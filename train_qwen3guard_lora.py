# train_qwen3guard_lora.py
"""使用 PEFT 对本地 Qwen3Guard-Gen-0.6B 进行 LoRA 微调（极速版）"""
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# ========== 1. 配置 ==========
MODEL_NAME = "./models/Qwen3Guard-Gen-0.6B"
DATA_PATH = "data/guard_train.json"
OUTPUT_DIR = "models/qwen3guard-lora-adapter"
MAX_LENGTH = 256  # 【修改点1】从 512 降到 256，速度翻倍

# ========== 2. 加载模型与分词器 ==========
print("正在加载本地模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# ========== 3. 配置 LoRA ==========
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========== 4. 加载并格式化数据 ==========
def format_example(example):
    messages = [
        {"role": "user", "content": f"{example['instruction']}\n\n待审核文本：{example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

print("正在加载训练数据...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 【修改点2】只取前 1000 条数据，极速验证
raw_data = raw_data[:1000]
print(f"截取训练数据量: {len(raw_data)} 条")

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_example, remove_columns=dataset.column_names)

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized_dataset = dataset.map(tokenize_fn, remove_columns=["text"])

# ========== 5. 训练参数 ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,                 # 数据少了，学习率恢复到 2e-4 让它学快点
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,           # 避免 Windows 下的多进程报错
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ========== 6. 开始训练 ==========
print("开始 LoRA 微调...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# ========== 7. 保存适配器 ==========
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA 适配器已保存至: {OUTPUT_DIR}")
# train_qwen3guard_lora.py
"""使用 PEFT 对本地 Qwen3Guard-Gen-0.6B 进行 LoRA 微调（极速版）"""
import json
import os
import sys
from pathlib import Path

# ---- 编码兜底（Windows 中文控制台）----
# cp950/cp936 下把 stdout 接到管道（如 Tee-Object）时，打印简体字会抛
# UnicodeEncodeError 直接中断训练。这里统一改成 utf-8 并把无法编码的字符替换掉，
# 保证「重定向日志」不会让脚本崩在第一条 print 上。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # 老版本解释器或被重定向到不支持的对象时忽略
        pass

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
MODEL_NAME = os.environ.get("QWEN_MODEL", "./models/Qwen3Guard-Gen-0.6B")
DATA_PATH = os.environ.get("LORA_DATA", "data/guard_train.json")
OUTPUT_DIR = os.environ.get("LORA_OUT", "models/qwen3guard-lora-adapter")
N_SAMPLES = int(os.environ.get("LORA_SAMPLES", "1000"))
MAX_STEPS = int(os.environ.get("LORA_MAX_STEPS", "-1"))
MAX_LENGTH = 256

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
raw_data = raw_data[:N_SAMPLES]
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
    max_steps=MAX_STEPS,
    logging_first_step=True,
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
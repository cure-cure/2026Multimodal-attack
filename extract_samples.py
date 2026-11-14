# extract_samples.py
import json


def extract(input_file, prefix):
    samples_ok, samples_fail = [], []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                # 兼容 C 模块输出的数据结构
                scores = r.get('component_scores', {})
                ocr_score = scores.get('ocr_text_score')

                if ocr_score is not None and len(samples_ok) < 3:
                    samples_ok.append(r)
                elif ocr_score is None and len(samples_fail) < 3:
                    samples_fail.append(r)

        with open(f'outputs/C_package/samples_normal_{prefix}.jsonl', 'w', encoding='utf-8') as f:
            for r in samples_ok: f.write(json.dumps(r, ensure_ascii=False) + '\n')

        with open(f'outputs/C_package/samples_ocr_failed_{prefix}.jsonl', 'w', encoding='utf-8') as f:
            for r in samples_fail: f.write(json.dumps(r, ensure_ascii=False) + '\n')

        print(f"[{prefix}] 成功提取 {len(samples_ok)} 条正常样例, {len(samples_fail)} 条 OCR 失败样例")
    except FileNotFoundError:
        print(f"未找到文件: {input_file}")


if __name__ == "__main__":
    # 提取自主分类器 (v2) 的结果
    extract('outputs/C_package/text_evidence.val.v2.jsonl', 'v2')
    # 提取 LoRA 模型的结果
    extract('outputs/C_package/text_evidence.val.lora.jsonl', 'lora')
"""运行D的完整模拟流程；只向outputs/D_mock写文件，不能报告为真实检测实验。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mm_jailbreak_guard.cli import main
from mm_jailbreak_guard.io import load_jsonl, write_jsonl


def run():
    folder = ROOT / 'outputs/D_mock'
    rows = load_jsonl(ROOT / 'examples/handoff/manifest.jsonl')
    for part in ('fusion_train', 'validation'):
        write_jsonl(folder / f'{part}.jsonl', [r for r in rows if r['split'] == part])
    baseline = str(ROOT / 'configs/fusion_config.example.json')
    evidence = ['--image-evidence', str(ROOT / 'examples/handoff/image_evidence.jsonl'),
                '--text-evidence', str(ROOT / 'examples/handoff/text_evidence.jsonl')]
    config = str(folder / 'learned.json')
    commands = [
        ['train-fusion', '--train-manifest', str(folder/'fusion_train.jsonl'),
         '--validation-manifest', str(folder/'validation.jsonl'), *evidence,
         '--baseline-config', baseline, '--output', config],
        ['compare', '--manifest', str(ROOT/'examples/handoff/manifest.jsonl'), *evidence,
         '--config', config, '--baseline-config', baseline, '--output', str(folder/'comparison.jsonl')],
        ['batch', '--manifest', str(ROOT/'examples/handoff/manifest.jsonl'), *evidence,
         '--config', config, '--output', str(folder/'predictions.jsonl')],
    ]
    for command in commands:
        code = main(command)
        if code:
            return code
    print('D模拟训练/比较/批量流程完成；结果不是模型成绩：'+str(folder))
    return 0


if __name__ == '__main__':
    raise SystemExit(run())

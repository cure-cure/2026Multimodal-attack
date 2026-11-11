"""Download only detection weights, with resumable HF cache and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "models"))
    p.add_argument("--revision", help="Optional LLaVA commit; otherwise resolve current main and record it")
    args = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = "llava-hf/llava-1.5-7b-hf"
    revision = HfApi().model_info(repo, revision=args.revision or "main").sha
    snapshot_download(repo_id=repo, revision=revision, local_dir=root / "llava-1.5-7b-hf",
                      allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.txt"])
    url = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
    target = root / "256x256_diffusion_uncond.pt"
    if not target.exists():
        temporary = target.with_suffix(".pt.partial")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    manifest = {"llava": {"repo": repo, "revision": revision}, "diffusion": {"url": url, "bytes": target.stat().st_size, "sha256": digest.hexdigest()}}
    (root / "WEIGHT_SOURCES.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

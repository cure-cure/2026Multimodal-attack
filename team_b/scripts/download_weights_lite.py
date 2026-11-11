"""Download only official tensors used by Task-B CIDER detection."""
import argparse
import hashlib
import json
from pathlib import Path

LLAVA_REPO = "llava-hf/llava-1.5-7b-hf"
DIFFUSION_REPO = "ydshieh/256x256_diffusion_uncond_from_blended_diffusion"
DIFFUSION_REVISION = "dfad65855ab5ac6efa536efd13c07f11b62d1433"
DIFFUSION_FILENAME = "256x256_diffusion_uncond.pt"
DIFFUSION_SHA256 = "a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "models"))
    parser.add_argument("--llava-revision", help="Commit; defaults to resolving current main")
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    llava_revision = HfApi().model_info(LLAVA_REPO, revision=args.llava_revision or "main").sha
    metadata = ["config.json", "generation_config.json", "preprocessor_config.json",
                "processor_config.json", "special_tokens_map.json", "tokenizer.json",
                "tokenizer.model", "tokenizer_config.json", "chat_template.json",
                "added_tokens.json", "model.safetensors.index.json"]
    snapshot_download(
        repo_id=LLAVA_REPO,
        revision=llava_revision,
        local_dir=root / "llava-1.5-7b-hf",
        allow_patterns=metadata + ["model-00001-of-00003.safetensors"],
    )
    downloaded = Path(hf_hub_download(
        repo_id=DIFFUSION_REPO,
        revision=DIFFUSION_REVISION,
        filename=DIFFUSION_FILENAME,
        local_dir=root,
    ))
    digest = sha256(downloaded)
    if digest != DIFFUSION_SHA256:
        raise RuntimeError(f"Diffusion SHA256 mismatch: {digest}")
    manifest = {
        "llava": {
            "repo": LLAVA_REPO,
            "revision": llava_revision,
            "mode": "detection_components",
            "shard": "model-00001-of-00003.safetensors",
            "used_prefixes": ["vision_tower", "multi_modal_projector",
                              "language_model.model.embed_tokens"],
        },
        "diffusion": {
            "canonical_source": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
            "download_repo": DIFFUSION_REPO,
            "revision": DIFFUSION_REVISION,
            "bytes": downloaded.stat().st_size,
            "sha256": digest,
        },
    }
    (root / "WEIGHT_SOURCES.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

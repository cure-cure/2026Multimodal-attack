"""B deployment entry point. Heavy dependencies are loaded only for inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
VERSION = "B-0.1.0"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_config(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    for key in ("encoder_path", "diffusion_path", "upstream_code"):
        value = Path(cfg[key])
        cfg[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    stages = cfg["stages"]
    if not stages or any(type(x) is not int or not 1 <= x <= 999 for x in stages):
        raise ValueError("stages must contain integer diffusion time indices in [1,999]")
    if stages != sorted(set(stages)):
        raise ValueError("stages must be unique and increasing")
    threshold = cfg.get("threshold")
    if threshold is not None and (type(threshold) not in (float, int) or not math.isfinite(threshold)):
        raise ValueError("threshold must be finite or null")
    if cfg.get("reproduction") != "adapted":
        raise ValueError("This deployment is an adapted implementation")
    return cfg


def statistic(similarities):
    if len(similarities) < 2 or any(not math.isfinite(x) for x in similarities):
        raise ValueError("Need finite original and denoised similarities")
    return max(similarities[0] - x for x in similarities[1:])


def rows(path):
    path = Path(path).resolve()
    seen = set()
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError(f"Invalid/duplicate sample_id at line {line_number}")
        seen.add(sid)
        if not isinstance(row.get("user_text"), str) or not row["user_text"].strip():
            raise ValueError(f"Missing user_text: {sid}")
        image = Path(row["image_path"])
        row["image_path"] = str(image if image.is_absolute() else (path.parent / image).resolve())
        result.append(row)
    if not result:
        raise ValueError("Empty manifest")
    return result


def dump_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


class Detector:
    def __init__(self, cfg):
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoTokenizer, AutoImageProcessor, LlavaForConditionalGeneration
        self.np, self.torch, self.Image = np, torch, Image
        self.cfg = cfg
        if not cfg["device"].startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("This diffusion deployment requires a CUDA GPU")
        for key in ("encoder_path", "diffusion_path", "upstream_code"):
            if not Path(cfg[key]).exists():
                raise FileNotFoundError(f"Missing {key}: {cfg[key]}")
        sys.path.insert(0, cfg["upstream_code"])
        from models.diffusion_denoiser.imagenet.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
        options = model_and_diffusion_defaults()
        options.update(image_size=256, num_channels=256, num_res_blocks=2,
                       num_heads=4, num_head_channels=64, attention_resolutions="32,16,8",
                       learn_sigma=True, class_cond=False, use_scale_shift_norm=True,
                       resblock_updown=True, use_fp16=False, diffusion_steps=1000,
                       noise_schedule="linear", timestep_respacing="")
        self.denoiser, self.diffusion = create_model_and_diffusion(**options)
        self.denoiser.load_state_dict(torch.load(cfg["diffusion_path"], map_location="cpu", weights_only=True))
        self.denoiser.eval()  # stays on CPU while the encoder is in GPU memory
        self.encoder = LlavaForConditionalGeneration.from_pretrained(
            cfg["encoder_path"], torch_dtype=torch.float16, low_cpu_mem_usage=True,
            local_files_only=True).eval()  # CPU initially; move by phase
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["encoder_path"], local_files_only=True)
        self.processor = AutoImageProcessor.from_pretrained(cfg["encoder_path"], local_files_only=True)

    def analyze(self, row):
        torch, np, Image = self.torch, self.np, self.Image
        cfg, device = self.cfg, self.cfg["device"]
        image_bytes = Path(row["image_path"]).read_bytes()
        # Seed by image content: repeated images get identical stages, independent of order.
        seed = (cfg["seed"] + int(hashlib.sha256(image_bytes).hexdigest()[:8], 16)) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        images = [image]
        self.encoder.cpu()
        torch.cuda.empty_cache()
        self.denoiser.to(device)
        x = torch.from_numpy(np.array(image).astype("float32") / 255 * 2 - 1).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.inference_mode():
            for stage in cfg["stages"]:
                t = torch.tensor([stage], device=device)
                noisy = self.diffusion.q_sample(x, t, noise=torch.randn_like(x))
                clean = self.diffusion.p_sample(self.denoiser, noisy, t, clip_denoised=True)["pred_xstart"]
                array = ((clean[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
                images.append(Image.fromarray(array))
        del x, noisy, clean
        self.denoiser.cpu()
        torch.cuda.empty_cache()
        self.encoder.to(device)
        similarities = []
        with torch.inference_mode():
            ids = self.tokenizer(row["user_text"], return_tensors="pt").input_ids.to(device)
            text = self.encoder.get_input_embeddings()(ids).mean(dim=1).float()
            for picture in images:
                pixels = self.processor(picture, return_tensors="pt").pixel_values.to(device, dtype=torch.float16)
                output = self.encoder.vision_tower(pixels, output_hidden_states=True)
                features = output.hidden_states[self.encoder.config.vision_feature_layer][:, 1:]
                visual = self.encoder.multi_modal_projector(features).mean(dim=1).float()
                similarities.append(torch.nn.functional.cosine_similarity(text, visual).item())
        value = statistic(similarities)
        threshold = cfg.get("threshold")
        return {"sample_id": row["sample_id"], "status": "success", "similarities": similarities,
                "stages": [0] + cfg["stages"], "statistic": value,
                "statistic_name": "max_original_minus_denoised", "threshold": threshold,
                "prediction": None if threshold is None else int(value > threshold),
                "comparison": "gt", "seed": seed, "config_id": cfg["config_id"],
                "reproduction": "adapted", "module_version": VERSION}


def main():
    parser = argparse.ArgumentParser(description="Independent CIDER detection; no answer generation")
    parser.add_argument("command", choices=["check", "batch", "single"])
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--manifest")
    parser.add_argument("--image")
    parser.add_argument("--text")
    parser.add_argument("--sample-id", default="demo-001")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "check":
        report = {"python": sys.version, "config": cfg, "paths": {k: Path(cfg[k]).exists() for k in ("encoder_path", "diffusion_path", "upstream_code")}}
        try:
            import torch
            report.update(torch=torch.__version__, cuda=torch.cuda.is_available())
            if torch.cuda.is_available():
                report["gpu"] = torch.cuda.get_device_name(0)
        except ImportError:
            report["torch"] = "not installed"
        if args.manifest:
            samples = rows(args.manifest)
            report["samples"] = len(samples)
            report["missing_images"] = [r["sample_id"] for r in samples if not Path(r["image_path"]).is_file()]
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if not args.output:
        parser.error("--output required")
    if args.command == "batch":
        if not args.manifest:
            parser.error("--manifest required")
        samples = rows(args.manifest)
    else:
        if not args.image or not args.text or not args.text.strip():
            parser.error("--image and nonempty --text required")
        samples = [{"sample_id": args.sample_id, "image_path": str(Path(args.image).resolve()), "user_text": args.text}]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Do not overwrite prior experimental results.
    if destination.exists():
        raise FileExistsError(destination)
    detector = Detector(cfg)  # setup failures must stop the run
    failures = 0
    with destination.open("x", encoding="utf-8") as stream:
        for row in samples:
            try:
                result = detector.analyze(row)
            except Exception as exc:
                failures += 1
                result = {"sample_id": row["sample_id"], "status": "failed", "error": f"{type(exc).__name__}: {exc}", "prediction": None, "module_version": VERSION}
            dump_line(stream, result)
            print(f"{row['sample_id']}: {result['status']}", flush=True)
    destination.with_suffix(destination.suffix + ".config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

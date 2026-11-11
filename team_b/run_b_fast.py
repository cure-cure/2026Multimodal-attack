"""A800-oriented CIDER runner: resident models, batched stages, image cache."""
import hashlib
import random
import time
from pathlib import Path

import run_b
from run_b_lite import LiteDetector


class FastLiteDetector(LiteDetector):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.denoiser.to(cfg["device"])
        self.encoder.to(cfg["device"])
        self.visual_cache = {}

    def analyze(self, row):
        torch, np, Image = self.torch, self.np, self.Image
        cfg, device = self.cfg, self.cfg["device"]
        started = time.perf_counter()
        image_bytes = Path(row["image_path"]).read_bytes()
        digest = hashlib.sha256(image_bytes).hexdigest()
        seed = (cfg["seed"] + int(digest[:8], 16)) % (2**32)
        cache_hit = digest in self.visual_cache
        if cache_hit:
            visuals = self.visual_cache[digest]
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            pictures = [image]
            x = torch.from_numpy(np.array(image).astype("float32") / 255 * 2 - 1)
            x = x.permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.inference_mode():
                for stage in cfg["stages"]:
                    t = torch.tensor([stage], device=device)
                    noisy = self.diffusion.q_sample(x, t, noise=torch.randn_like(x))
                    clean = self.diffusion.p_sample(
                        self.denoiser, noisy, t, clip_denoised=True
                    )["pred_xstart"]
                    array = ((clean[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5)
                    pictures.append(Image.fromarray(array.clip(0, 255).astype("uint8")))
                pixels = self.processor(pictures, return_tensors="pt").pixel_values
                pixels = pixels.to(device, dtype=torch.float16)
                output = self.encoder.vision_tower(pixels, output_hidden_states=True)
                features = output.hidden_states[self.encoder.config.vision_feature_layer][:, 1:]
                visuals = self.encoder.multi_modal_projector(features).mean(dim=1).float().cpu()
            self.visual_cache[digest] = visuals

        with torch.inference_mode():
            ids = self.tokenizer(row["user_text"], return_tensors="pt").input_ids.to(device)
            text = self.encoder.get_input_embeddings()(ids).mean(dim=1).float()
            similarities = torch.nn.functional.cosine_similarity(
                text.expand(len(visuals), -1), visuals.to(device), dim=1
            ).cpu().tolist()
        value = run_b.statistic(similarities)
        threshold = cfg.get("threshold")
        return {
            "sample_id": row["sample_id"], "status": "success",
            "similarities": similarities, "stages": [0] + cfg["stages"],
            "statistic": value, "statistic_name": "max_original_minus_denoised",
            "threshold": threshold,
            "prediction": None if threshold is None else int(value > threshold),
            "comparison": "gt", "seed": seed, "image_sha256": digest,
            "visual_cache_hit": cache_hit, "config_id": cfg["config_id"],
            "reproduction": "adapted", "module_version": "B-1.0.0-fast-lite",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }


if __name__ == "__main__":
    run_b.Detector = FastLiteDetector
    run_b.VERSION = "B-1.0.0-fast-lite"
    run_b.main()

"""CIDER entry using only LLaVA components actually touched by detection.

The causal language-model decoder is never called by CIDER.  This loader reads
the CLIP vision tower, multimodal projector, and input embedding table directly
from the official first LLaVA safetensors shard, reducing required download and
CPU memory while preserving the exact tensors used in the similarity formula.
"""
from pathlib import Path
import sys

import run_b


class LiteDetector(run_b.Detector):
    def __init__(self, cfg):
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoTokenizer, AutoImageProcessor

        self.np, self.torch, self.Image = np, torch, Image
        self.cfg = cfg
        if not cfg["device"].startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("This diffusion deployment requires a CUDA GPU")
        shard = Path(cfg["encoder_path"]) / "model-00001-of-00003.safetensors"
        for key in ("encoder_path", "diffusion_path", "upstream_code"):
            if not Path(cfg[key]).exists():
                raise FileNotFoundError(f"Missing {key}: {cfg[key]}")
        if not shard.is_file():
            raise FileNotFoundError(f"Missing detection-component shard: {shard}")

        sys.path.insert(0, cfg["upstream_code"])
        from models.diffusion_denoiser.imagenet.guided_diffusion.script_util import (
            create_model_and_diffusion, model_and_diffusion_defaults,
        )
        options = model_and_diffusion_defaults()
        options.update(image_size=256, num_channels=256, num_res_blocks=2,
                       num_heads=4, num_head_channels=64, attention_resolutions="32,16,8",
                       learn_sigma=True, class_cond=False, use_scale_shift_norm=True,
                       resblock_updown=True, use_fp16=False, diffusion_steps=1000,
                       noise_schedule="linear", timestep_respacing="")
        self.denoiser, self.diffusion = create_model_and_diffusion(**options)
        self.denoiser.load_state_dict(
            torch.load(cfg["diffusion_path"], map_location="cpu", weights_only=True)
        )
        self.denoiser.eval()
        self.encoder = self._load_detection_encoder(Path(cfg["encoder_path"]), shard)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["encoder_path"], local_files_only=True, use_fast=False)
        self.processor = AutoImageProcessor.from_pretrained(cfg["encoder_path"], local_files_only=True)

    def _load_detection_encoder(self, metadata_path, shard_path):
        torch = self.torch
        from safetensors import safe_open
        from transformers import CLIPVisionConfig, CLIPVisionModel

        config = run_b.read_json(metadata_path / "config.json")
        vision = CLIPVisionModel(CLIPVisionConfig(**config["vision_config"])).half()
        hidden_size = int(config.get("text_config", {}).get("hidden_size", 4096))
        vocab_size = int(config.get("text_config", {}).get("vocab_size", config["vocab_size"]))

        class Projector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_1 = torch.nn.Linear(config["vision_config"]["hidden_size"], hidden_size)
                self.act = torch.nn.GELU()
                self.linear_2 = torch.nn.Linear(hidden_size, hidden_size)

            def forward(self, value):
                return self.linear_2(self.act(self.linear_1(value)))

        class DetectionEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = vision
                self.multi_modal_projector = Projector().half()
                self.input_embeddings = torch.nn.Embedding(vocab_size, hidden_size).half()
                self.config = type("DetectionConfig", (), {
                    "vision_feature_layer": int(config.get("vision_feature_layer", -2))
                })()

            def get_input_embeddings(self):
                return self.input_embeddings

        encoder = DetectionEncoder()
        with safe_open(shard_path, framework="pt", device="cpu") as source:
            vision_state = {
                key.removeprefix("vision_tower."): source.get_tensor(key)
                for key in source.keys() if key.startswith("vision_tower.")
            }
            projector_state = {
                key.removeprefix("multi_modal_projector."): source.get_tensor(key)
                for key in source.keys() if key.startswith("multi_modal_projector.")
            }
            embedding = source.get_tensor("language_model.model.embed_tokens.weight")
        encoder.vision_tower.load_state_dict(vision_state, strict=True)
        encoder.multi_modal_projector.load_state_dict(projector_state, strict=True)
        with torch.no_grad():
            encoder.input_embeddings.weight.copy_(embedding)
        return encoder.eval()


if __name__ == "__main__":
    run_b.Detector = LiteDetector
    run_b.VERSION = "B-1.0.0-lite"
    run_b.main()

"""OpenAI CLIP via OpenCLIP; no training, gradients or probability claims."""
import hashlib
import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F


def select_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or auto")
    torch.device(requested)  # validate syntax
    return requested


class CLIPClassifier:
    def __init__(self, classes, model_name, pretrained, prompt, device, cache_dir):
        self.device = select_device(device)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Use the canonical OpenAI checkpoint and OpenCLIP's SHA256 verification.
        # Other tags retain OpenCLIP's default download behavior.
        self.weight_path = None
        checkpoint = pretrained
        if pretrained == "openai":
            self.weight_path = open_clip.download_pretrained(
                open_clip.get_pretrained_cfg(model_name, pretrained), prefer_hf_hub=False,
                cache_dir=str(cache_dir / "weights"))
            checkpoint = self.weight_path
        # OpenAI weights require QuickGELU, rather than the default GELU configuration.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=checkpoint, device=self.device,
            force_quick_gelu=True, cache_dir=str(cache_dir / "weights"),
            # The SHA256-verified official OpenAI file is a TorchScript archive.
            # PyTorch >=2.6 cannot load this archive with weights_only=True.
            weights_only=pretrained != "openai")
        self.model.eval()
        self.model.requires_grad_(False)
        self.prompts = [prompt.format(breed=name) for name in classes]
        key = json.dumps({"model": model_name, "pretrained": pretrained,
                          "prompts": self.prompts, "open_clip": open_clip.__version__,
                          "quick_gelu": True}, sort_keys=True)
        self.text_cache_key = hashlib.sha256(key.encode()).hexdigest()
        text_path = cache_dir / f"text_{self.text_cache_key}.pt"
        with torch.inference_mode():
            if text_path.exists():
                text = torch.load(text_path, weights_only=True, map_location="cpu")
            else:
                tokens = open_clip.get_tokenizer(model_name)(self.prompts).to(self.device)
                text = F.normalize(self.model.encode_text(tokens).float(), dim=-1).cpu()
                torch.save(text, text_path)
            if text.shape[0] != len(classes) or not torch.isfinite(text).all():
                raise RuntimeError("Invalid cached text embeddings")
            self.text = text.to(self.device)

    @torch.inference_mode()
    def predict(self, tensors):
        images = torch.stack(tensors).to(self.device)
        features = F.normalize(self.model.encode_image(images).float(), dim=-1)
        similarities = features @ self.text.T
        values, predictions = similarities.max(dim=-1)
        return predictions.cpu().numpy(), values.cpu().numpy()

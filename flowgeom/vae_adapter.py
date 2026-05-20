from __future__ import annotations
import warnings
import torch
import torch.nn.functional as F


class VAEAdapter:
    def __init__(self, mode: str = "identity", device: str = "cpu"):
        self.mode = mode
        self.device = device
        self.vae = None
        if mode == "diffusers_vae":
            try:
                from diffusers import AutoencoderKL  # type: ignore
                self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
                self.vae.eval()
            except Exception:
                warnings.warn("Diffusers VAE unavailable; fallback to downsample latent.")
                self.mode = "identity"

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        x = image_tensor.to(self.device)
        if self.mode == "identity":
            return F.interpolate(x.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False).squeeze(0)
        with torch.no_grad():
            lat = self.vae.encode(x.unsqueeze(0)).latent_dist.sample() * 0.18215
        return lat.squeeze(0)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return F.interpolate(z.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).squeeze(0)
        with torch.no_grad():
            img = self.vae.decode((z.unsqueeze(0) / 0.18215)).sample
        return img.squeeze(0)

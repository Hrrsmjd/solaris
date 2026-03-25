from datetime import timedelta

import torch

from solaris.model.decoder import Perceiver3DDecoder
from solaris.model.encoder import Perceiver3DEncoder
from solaris.model.lora import LoRAMode
from solaris.model.swin3d import Swin3DTransformerBackbone

__all__ = ["Solaris", "SolarisHighRes", "SolarisSmall"]


class Solaris(torch.nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 512,
        encoder_depths: tuple[int, ...] = (6, 10, 8),
        encoder_num_heads: tuple[int, ...] = (8, 16, 32),
        decoder_depths: tuple[int, ...] = (8, 10, 6),
        decoder_num_heads: tuple[int, ...] = (32, 16, 8),
        window_size: tuple[int, int, int] = (2, 6, 12),
        max_history_size: int = 2,
        latent_levels: int = 4,
        out_levels: int = 9,
        enc_depth: int = 1,
        dec_depth: int = 1,
        num_heads: int = 16,
        drop_rate: float = 0.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        dec_mlp_ratio: float = 2.0,
        perceiver_ln_eps: float = 1e-5,
        use_lora: bool = True,
        lora_mode: LoRAMode = "single",
        lora_steps: int = 40,
    ):
        super().__init__()
        self.out_levels = out_levels

        self.encoder = Perceiver3DEncoder(
            patch_size=patch_size,
            embed_dim=embed_dim,
            max_history_size=max_history_size,
            latent_levels=latent_levels,
            depth=enc_depth,
            head_dim=embed_dim // num_heads,
            num_heads=num_heads,
            drop_rate=drop_rate,
            mlp_ratio=mlp_ratio,
            perceiver_ln_eps=perceiver_ln_eps,
        )

        self.backbone = Swin3DTransformerBackbone(
            embed_dim=embed_dim,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path,
            use_lora=use_lora,
            lora_mode=lora_mode,
            lora_steps=lora_steps,
        )

        self.decoder = Perceiver3DDecoder(
            patch_size=patch_size,
            embed_dim=embed_dim * 2,
            out_levels=out_levels,
            depth=dec_depth,
            head_dim=embed_dim * 2 // num_heads,
            num_heads=num_heads,
            drop_rate=drop_rate,
            mlp_ratio=dec_mlp_ratio,
            perceiver_ln_eps=perceiver_ln_eps,
        )

        self.register_buffer("means", torch.zeros(out_levels))
        self.register_buffer("stds", torch.ones(out_levels))

    def set_normalisation(self, means, stds):
        assert means.shape == (self.out_levels,) and stds.shape == (self.out_levels,)
        self.means.copy_(means)
        self.stds.copy_(stds)

    def normalise(self, x):
        if self.means.numel() != x.shape[2]:
            return x
        means = self.means.view(1, 1, -1, 1, 1)
        stds = self.stds.view(1, 1, -1, 1, 1)
        return (x - means) / stds

    def unnormalise(self, x):
        if self.means.numel() != x.shape[2]:
            return x
        means = self.means.view(1, 1, -1, 1, 1)
        stds = self.stds.view(1, 1, -1, 1, 1)
        return (x * stds) + means

    def forward(self, x, metadata, lead_time, rollout_step):
        lead_time = lead_time if isinstance(lead_time, timedelta) else timedelta(hours=lead_time)
        x = self.normalise(x)

        H, W = x.size(-2), x.size(-1)
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )

        x = self.encoder(x, metadata, lead_time)

        x = self.backbone(x, lead_time, rollout_step=rollout_step, patch_res=patch_res)

        x = self.decoder(x, metadata, lead_time, patch_res=patch_res)

        x = self.unnormalise(x)

        return x


Solaris = Solaris
SolarisHighRes = Solaris
SolarisSmall = Solaris

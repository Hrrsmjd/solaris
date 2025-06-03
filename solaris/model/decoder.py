import torch
import torch.nn as nn
from einops import rearrange

from solaris.model.perceiver import PerceiverResampler
from solaris.model.util import init_weights, unpatchify

__all__ = ["Perceiver3DDecoder"]


class Perceiver3DDecoder(nn.Module):
    """Multi-scale multi-source multi-variable decoder based on the Perceiver architecture."""

    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 1024,
        out_levels: int = 8,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 16,
        drop_rate: float = 0.1,
        mlp_ratio: float = 4.0,
        perceiver_ln_eps: float = 1e-5,  ##
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        assert out_levels > 1, "At least two latent levels are required."
        self.out_levels = out_levels
        self.latents = nn.Parameter(torch.randn(out_levels, embed_dim))

        self.level_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
            mlp_ratio=mlp_ratio,
            ln_eps=perceiver_ln_eps,
        )

        self.x_head = nn.Linear(embed_dim, patch_size**2)

        self.apply(init_weights)
        torch.nn.init.trunc_normal_(self.latents, std=0.02)

    def deaggregate_levels(self, x):
        B, L, C, D = x.shape
        latents = self.latents.to(dtype=x.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)  # (C_A, D) to (B, C_A, L, D)

        latents = torch.einsum("b c l d -> b l c d", latents)
        latents = latents.flatten(0, 1)  # (B * L, C, D)
        x = x.flatten(0, 1)  # (B * L, C, D)

        x = self.level_agg(latents, x)  # (B * L, C, D)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, C, D)
        # x = torch.einsum("b l c d -> b c l d", x)
        return x

    def forward(self, x, metadata, lead_time, patch_res):
        # Compress the latent dimension from the U-net skip concatenation.
        B, L, D = x.shape

        # Extract pix_x, pix_y and convert to float32.
        pix_x, pix_y = metadata[0], metadata[1]
        pix_x, pix_y = pix_x.to(dtype=torch.float32), pix_y.to(dtype=torch.float32)
        H, W = pix_x.shape[0], pix_y.shape[-1]

        # Unwrap the latent level dimension.
        x = rearrange(
            x,
            "B (C H W) D -> B (H W) C D",
            C=patch_res[0],
            H=patch_res[1],
            W=patch_res[2],
        )

        # De-aggregate the hidden levels into the physical levels.
        x = self.deaggregate_levels(x)

        # Decode the wavelength levels.
        x = self.x_head(x)
        x = unpatchify(x, 1, H, W, self.patch_size)

        return x

import torch
import torch.nn as nn
from einops import rearrange

from solaris.model.fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    pos_expansion_for_the_sun,
)
from solaris.model.patchembed import LevelPatchEmbed
from solaris.model.perceiver import PerceiverResampler
from solaris.model.posencoding import pos_enc
from solaris.model.util import init_weights

__all__ = ["Perceiver3DEncoder"]


class Perceiver3DEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 1024,
        max_history_size: int = 2,
        latent_levels: int = 8,
        depth: int = 2,
        head_dim: int = 64,
        num_heads: int = 16,
        drop_rate: float = 0.1,
        mlp_ratio: float = 4.0,
        perceiver_ln_eps: float = 1e-5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.max_history_size = max_history_size

        self.x_token_embeds = LevelPatchEmbed(patch_size, embed_dim, max_history_size)

        assert latent_levels > 1, "At least two latent levels are required."
        self.latent_levels = latent_levels
        self.latents = nn.Parameter(torch.randn(latent_levels, embed_dim))

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

        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(init_weights)
        torch.nn.init.trunc_normal_(self.latents, std=0.02)

    def aggregate_levels(self, x):
        B, C, L, D = x.shape
        latents = self.latents.to(dtype=x.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)  # (C_A, D) to (B, C_A, L, D)

        latents = torch.einsum("b c l d -> b l c d", latents)
        latents = latents.flatten(0, 1)  # (B * L, C, D)
        x = torch.einsum("b c l d -> b l c d", x)
        x = x.flatten(0, 1)  # (B * L, C, D)

        x = self.level_agg(latents, x)  # (B * L, C, D)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, C, D)
        x = torch.einsum("b l c d -> b c l d", x)
        return x

    def forward(self, x, metadata, lead_time):
        B, T, C, H, W = x.shape
        pix_x, pix_y, _ = metadata
        if pix_x.shape[0] != H or pix_y.shape[-1] != W:
            raise ValueError(
                "Metadata spatial dimensions do not match the input tensor: "
                f"expected ({H}, {W}), got ({pix_x.shape[0]}, {pix_y.shape[-1]})."
            )

        # Patch embed the wavelength levels.
        x = rearrange(x, "b t c h w -> (b c) 1 t h w")
        x = self.x_token_embeds(x)
        x = rearrange(x, "(b c) l d -> b c l d", b=B, c=C)
        dtype = x.dtype  # When using mixed precision, we need to keep track of the dtype.

        # Aggregate over wavelength levels.
        x = self.aggregate_levels(x)

        # Add position embeddings to the 3D tensor.
        pos_encode = pos_enc(
            self.embed_dim,
            pix_x,
            pix_y,
            self.patch_size,
            pos_expansion=pos_expansion_for_the_sun,
        )
        # Encodings are (L, D).
        pos_encode = self.pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        x = x + pos_encode

        # Flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C, L, D) to (B, L', D)

        # Add absolute time embedding.
        absolute_times_list = [t.timestamp() / 3600 for t in metadata[2]]  # Times in hours
        absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
        absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        # Add lead time embedding.
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        x = self.pos_drop(x)
        return x

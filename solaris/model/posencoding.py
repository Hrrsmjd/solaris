import torch
import torch.nn.functional as F

from solaris.model.fourier import FourierExpansion

__all__ = ["pos_enc"]


def pix_x_y_meshgrid(pix_x: torch.Tensor, pix_y: torch.Tensor):
    assert pix_x.dim() == pix_y.dim() == 1

    grid = torch.meshgrid(pix_x, pix_y, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.permute(0, 2, 1)

    return grid


def pos_enc_grid(
    encode_dim: int,
    grid: torch.Tensor,
    patch_dims: int | list | tuple,
    pos_expansion: FourierExpansion,
):
    assert encode_dim % 4 == 0
    assert grid.dim() == 4

    # Take the 2D pooled values of the mesh. This is the same as subsequent
    # 1D pooling over the x-axis and then over the y-axis.
    grid_h = F.avg_pool2d(grid[:, 0], patch_dims)
    grid_w = F.avg_pool2d(grid[:, 1], patch_dims)

    # Use half of the dimensions for pix_x of the midpoints of the patches
    # and the other half for pix_y.
    # Before computing the encodings, flatten over the spatial dimensions.
    B = grid_h.shape[0]
    encode_h = pos_expansion(grid_h.reshape(B, -1), encode_dim // 2)  # (B, L, D/2)
    encode_w = pos_expansion(grid_w.reshape(B, -1), encode_dim // 2)  # (B, L, D/2)
    pos_encode = torch.cat((encode_h, encode_w), axis=-1)  # (B, L, D)

    return pos_encode


def pos_enc(
    encode_dim: int,
    pix_x: torch.Tensor,
    pix_y: torch.Tensor,
    patch_dims: int | list | tuple,
    pos_expansion: FourierExpansion,
):
    if pix_x.dim() == pix_y.dim() == 1:
        grid = pix_x_y_meshgrid(pix_x, pix_y)
    elif pix_x.dim() == pix_y.dim() == 2:
        grid = torch.stack((pix_x, pix_y), dim=0)
    else:
        raise ValueError(
            f"Pix_x and pix_y must either both be vectors or both be matrices, "
            f"but have dimensionalities {pix_x.dim()} and {pix_y.dim()} respectively."
        )

    grid = grid[None]  # Add batch dimension.

    pos_encoding = pos_enc_grid(encode_dim, grid, patch_dims, pos_expansion)

    return pos_encoding.squeeze(0)  # Return without batch dimension.

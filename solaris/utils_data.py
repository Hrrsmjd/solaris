from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch

AIA_INPUT_WAVELENGTHS = ("0094", "0131", "0171", "0193", "0304", "0335")


def parse_custom_hour(hour_str: str) -> int:
    """Convert custom hour format ``H0000`` into an integer hour."""
    return int(hour_str[1:3])


def to_custom_hour(hour: int) -> str:
    """Convert an integer hour into the custom ``H0000`` format."""
    return f"H{hour:02d}00"


def add_hours(date_time_list: Iterable[str], hours_to_add: int) -> list[str]:
    """Offset a timestamp expressed as ``[year, month, day, hour]``."""
    year, month, day, hour_str = date_time_list
    hour = parse_custom_hour(hour_str)

    original_datetime = datetime(int(year), int(month), int(day), hour)
    new_datetime = original_datetime + timedelta(hours=hours_to_add)

    return [
        str(new_datetime.year),
        f"{new_datetime.month:02d}",
        f"{new_datetime.day:02d}",
        to_custom_hour(new_datetime.hour),
    ]


def resolve_data_root(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the directory containing Solaris HDF5 data files."""
    candidate = path if path not in (None, "", "CHANGE_PATH") else os.environ.get("SOLARIS_DATA_DIR")
    if not candidate:
        raise ValueError(
            "Data path is not configured. Pass an explicit path or set SOLARIS_DATA_DIR."
        )
    return Path(candidate).expanduser()


def resolve_id_dir(
    id_dir: str | os.PathLike[str] | None = None,
    *,
    data_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the directory containing train/val/test ID files."""
    candidate = id_dir if id_dir not in (None, "", "CHANGE_PATH") else os.environ.get("SOLARIS_ID_DIR")
    if candidate:
        return Path(candidate).expanduser()
    return resolve_data_root(data_root)


def read_id_file(path: str | os.PathLike[str]) -> list[list[str]]:
    """Read whitespace-separated timestamp IDs from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        return [line.split() for line in file if line.strip()]


def load_wavelength_stack(
    root_dir: str | os.PathLike[str],
    timestamp: Iterable[str],
    wavelengths: Iterable[str] = AIA_INPUT_WAVELENGTHS,
) -> torch.Tensor:
    """Load a stack of wavelength channels for a single timestamp."""
    year, month, day, hour = timestamp
    root_path = resolve_data_root(root_dir)
    with h5py.File(root_path / f"{year}.h5", "r") as file:
        channels = [
            torch.from_numpy(
                np.asarray(file[year][month][day][hour][wavelength], dtype=np.float32)
            )[None, ...]
            for wavelength in wavelengths
        ]
    return torch.cat(channels, dim=0)


def load_target_channel(
    root_dir: str | os.PathLike[str],
    timestamp: Iterable[str],
    wavelength: str = "1700",
) -> torch.Tensor:
    """Load a single target wavelength channel for a timestamp."""
    year, month, day, hour = timestamp
    root_path = resolve_data_root(root_dir)
    with h5py.File(root_path / f"{year}.h5", "r") as file:
        return torch.from_numpy(
            np.asarray(file[year][month][day][hour][wavelength], dtype=np.float32)
        )[None, ...]


def build_metadata(batch: torch.Tensor, timestamp: datetime | None = None):
    """Construct Solaris metadata from a batch of shape `(B, C, H, W)`."""
    if batch.dim() != 4:
        raise ValueError(f"Expected a 4D batch `(B, C, H, W)`, got shape {tuple(batch.shape)}.")

    _, _, height, width = batch.shape
    reference_time = timestamp or datetime(1970, 1, 1)

    return (
        torch.arange(height, device=batch.device, dtype=torch.float32),
        torch.arange(width, device=batch.device, dtype=torch.float32),
        tuple(reference_time for _ in range(batch.shape[0])),
    )

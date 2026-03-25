from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import hf_hub_download

from solaris.utils_data import add_hours, resolve_data_root, resolve_id_dir

YEARS = tuple(str(year) for year in range(2010, 2024))
DATASET_REPO_ID = "hrrsmjd/AIA_12hour_512x512"
FILE_PREFIX = "aia_12hour_512x512_"


def _data_file(root_dir: str | Path, year: str) -> Path:
    return resolve_data_root(root_dir) / f"{FILE_PREFIX}{year}.h5"


def download_year(year: str, output_dir: str | Path | None = None) -> Path:
    """Download a single year of processed Solaris data."""
    target_dir = resolve_data_root(output_dir)
    return Path(
        hf_hub_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            filename=f"{FILE_PREFIX}{year}.h5",
            local_dir=target_dir,
        )
    )


def check_data_exists(root_dir: str | Path, year: str = "2023") -> None:
    """Print the number of datasets flagged as present for a given year."""
    with h5py.File(_data_file(root_dir, year), "r") as file:
        count = 0
        for year_key in file.keys():
            for month_key in file[year_key].keys():
                for day_key in file[year_key][month_key].keys():
                    for hour_key in file[year_key][month_key][day_key].keys():
                        for wavelength_key in file[year_key][month_key][day_key][hour_key].keys():
                            dataset = file[year_key][month_key][day_key][hour_key][wavelength_key]
                            if "exists" in dataset.attrs and dataset.attrs["exists"]:
                                count += 1
        print(f"Total existing data points in {year}: {count}")


def get_valid_ids_for_downstream_task(root_dir: str | Path) -> list[list[str]]:
    """Return timestamps whose 12-hour target exists for the downstream task."""
    valid_ids = []

    for year in ("2019", "2020", "2021", "2022", "2023"):
        with h5py.File(_data_file(root_dir, year), "r") as file:
            for year_key in file.keys():
                for month_key in file[year_key].keys():
                    for day_key in file[year_key][month_key].keys():
                        for hour_key in file[year_key][month_key][day_key].keys():
                            timestamp = [year_key, month_key, day_key, hour_key]
                            future_timestamp = add_hours(timestamp, 12)
                            try:
                                with h5py.File(_data_file(root_dir, future_timestamp[0]), "r") as future_file:
                                    exists = future_file[future_timestamp[0]][future_timestamp[1]][
                                        future_timestamp[2]
                                    ][future_timestamp[3]]["1700"].attrs["exists"]
                                    if exists:
                                        valid_ids.append(timestamp)
                            except (OSError, KeyError):
                                continue

    return valid_ids


def save_downstream_ids(root_dir: str | Path, output_file: str | Path) -> None:
    """Persist valid downstream timestamps to disk."""
    valid_ids = get_valid_ids_for_downstream_task(root_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for id_list in valid_ids:
            file.write(" ".join(id_list) + "\n")
    print(f"Saved {len(valid_ids)} valid IDs to {output_path}")


def save_the_id_downstram(path, path_id):
    """Backward-compatible alias for the original misspelled function."""
    save_downstream_ids(path, path_id)


def save_pretrain_ids(root_dir: str | Path, output_dir: str | Path) -> None:
    """Create train/val/test split files for pretraining."""
    all_ids = []

    for year in ("2019", "2020", "2021", "2022", "2023"):
        with h5py.File(_data_file(root_dir, year), "r") as file:
            for year_key in file.keys():
                for month_key in file[year_key].keys():
                    for day_key in file[year_key][month_key].keys():
                        for hour_key in file[year_key][month_key][day_key].keys():
                            all_ids.append([year_key, month_key, day_key, hour_key])

    np.random.shuffle(all_ids)

    train_split = int(0.8 * len(all_ids))
    val_split = int(0.9 * len(all_ids))
    splits = {
        "train": all_ids[:train_split],
        "val": all_ids[train_split:val_split],
        "test": all_ids[val_split:],
    }

    output_root = resolve_id_dir(output_dir, data_root=root_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for split, ids in splits.items():
        with (output_root / f"{split}_id.txt").open("w", encoding="utf-8") as file:
            for id_list in ids:
                file.write(" ".join(id_list) + "\n")
        print(f"Saved {len(ids)} {split} IDs")


def save_the_id_pretrain(path, path_id):
    """Backward-compatible alias for the original function name."""
    save_pretrain_ids(path, path_id)

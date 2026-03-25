from pathlib import Path

import h5py

from solaris.utils_data import resolve_data_root


def update_exists_attribute(file_path, year, month, day, hour, wavelength):
    """Set the ``exists`` attribute to ``False`` for a specific dataset."""
    with h5py.File(file_path, "r+") as hdf5_file:
        dataset = hdf5_file[year][month][day][hour][wavelength]
        if "exists" in dataset.attrs and dataset.attrs["exists"]:
            dataset.attrs["exists"] = False
            print(
                f"Successfully changed 'exists' attribute to False for "
                f"{year}/{month}/{day}/{hour}/{wavelength}"
            )
        else:
            print(
                f"'exists' attribute is either not present or already False for "
                f"{year}/{month}/{day}/{hour}/{wavelength}"
            )


FILES_TO_PROCESS = [
    ("aia_12hour_512x512_2019.h5", [("2019", "01", "13", "H0000", "0304")]),
    (
        "aia_12hour_512x512_2021.h5",
        [("2021", "04", "29", "H1200", wavelength) for wavelength in ("0094", "0131", "0171", "0193", "0304", "0335")],
    ),
    ("aia_12hour_512x512_2022.h5", [("2022", "02", "04", "H0000", "0211")]),
]


def apply_known_fixes(data_directory: str | Path) -> None:
    """Apply the curated list of known HDF5 metadata fixes."""
    root = resolve_data_root(data_directory)
    for filename, dataset_list in FILES_TO_PROCESS:
        file_path = root / filename
        if file_path.exists():
            print(f"Processing file: {filename}")
            for year, month, day, hour, wavelength in dataset_list:
                update_exists_attribute(file_path, year, month, day, hour, wavelength)
        else:
            print(f"File not found: {file_path}")
    print("Processing complete.")

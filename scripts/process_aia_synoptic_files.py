import argparse
import os
from datetime import datetime

import astropy.units as u
import h5py
import numpy as np
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_correction_table
from scipy.ndimage import zoom
from sunpy.map import GenericMap, Map


def degradation_correction(X):
    """Apply degradation corrections to map X."""
    return correct_degradation(X, correction_table=get_correction_table())


def exposure_correction(X):
    """Apply exposure time corrections to map X."""
    X_norm = X / X.exposure_time
    X_norm.meta["EXPTIME"] = 1
    return X_norm


def resize_to_1024(X):
    """Resize the input to 1024x1024."""
    h, w = X.shape
    if h == 1024 and w == 1024:
        return X

    X_resize = np.zeros((1024, 1024))
    if h > 1024 or w > 1024:
        start_y = max(0, (h - 1024) // 2)
        start_x = max(0, (w - 1024) // 2)
        crop = X[start_y : start_y + 1024, start_x : start_x + 1024]
        X_resize[0 : crop.shape[0], 0 : crop.shape[1]] = crop
    else:
        start_y = (1024 - h) // 2
        start_x = (1024 - w) // 2
        X_resize[start_y : start_y + h, start_x : start_x + w] = X

    return X_resize


def scale_and_center(X, target_angular_size=976.0, target_size=512):
    """Scale and center the input map data to a target angular size."""
    if 1024 % target_size != 0:
        raise ValueError("target_size must be a proper divisor of 1024")

    X_data = X.data
    X_meta = X.meta

    scale_factor = target_angular_size / X.rsun_obs.value
    X_zoom = zoom(X_data, scale_factor)
    X_meta["RSUN_OBS"] = target_angular_size

    X_zoom[X_zoom < 0] = 0
    X_resize = resize_to_1024(X_zoom)

    X = GenericMap(X_resize, X_meta)
    return X.resample([target_size, target_size] * u.pixel)


def process_aia_file(file_path, wavelength):
    """Process a single AIA FITS file."""
    try:
        aia_map = Map(file_path)
        if aia_map.meta["QUALITY"] != 0:
            raise ValueError("Quality is not 0.")

        if wavelength != "4500":
            aia_map = degradation_correction(aia_map)
        aia_map = exposure_correction(aia_map)
        aia_map = scale_and_center(aia_map)

        data = aia_map.data.astype(np.float32)

        # Convert the observation time to a datetime object
        obs_time = datetime.strptime(aia_map.meta["DATE-OBS"], "%Y-%m-%dT%H:%M:%S.%f")

        meta = {
            "filename": os.path.basename(file_path),
            "datetime": obs_time,
            "exists": True,
            "size": data.size,
            "sum": data.sum(),
            "sum_squared": (data**2).sum(),
        }
        return data, meta
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None


def process_aia_directory(source_path, target_path):
    """Process all AIA files in a directory and its subdirectories."""
    wavelengths = ["0094", "0131", "0171", "0193", "0211", "0304", "0335", "1600", "1700", "4500"]

    with h5py.File(target_path, "a") as hf:
        for dirpath, _, filenames in os.walk(source_path):
            fits_files = [f for f in filenames if f.startswith("AIA") and f.endswith(".fits")]
            if not fits_files:
                continue

            year, month, day, time = (
                fits_files[0][3:7],
                fits_files[0][7:9],
                fits_files[0][9:11],
                fits_files[0][12:16],
            )

            group_path = f"{year}/{month}/{day}/H{time}"

            # Check if this group already exists and has all wavelengths
            if group_path in hf and all(wavelength in hf[group_path] for wavelength in wavelengths):
                print(f"Skipping existing data: {group_path}")
                continue

            group = hf.require_group(group_path)

            for wavelength in wavelengths:
                # Skip if this wavelength already exists in the group
                if wavelength in group:
                    print(f"Skipping existing wavelength: {group_path}/{wavelength}")
                    continue

                filename = f"AIA{year}{month}{day}_{time}_{wavelength}.fits"
                file_path = os.path.join(dirpath, filename)

                if os.path.exists(file_path):
                    data, meta = process_aia_file(file_path, wavelength)
                else:
                    data, meta = None, None

                if data is None or meta is None:
                    data = np.full((512, 512), np.nan, dtype=np.float32)
                    meta = {
                        "filename": filename,
                        "datetime": datetime(
                            int(year), int(month), int(day), int(time[:2]), int(time[2:])
                        ),
                        "exists": False,
                        "size": data.size,
                        "sum": np.nan,
                        "sum_squared": np.nan,
                    }

                dataset = group.create_dataset(wavelength, data=data)
                for key, value in meta.items():
                    if key == "datetime":
                        # Store datetime as a string attribute
                        dataset.attrs[key] = value.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        dataset.attrs[key] = value

                print(f"Processed: {group_path}/{wavelength}")

        # Ensure all data is written to disk
        hf.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AIA synoptic files.")
    parser.add_argument("--source_path", help="Path to the source directory containing AIA files")
    parser.add_argument("--target_path", help="Path to the target HDF5 file for processed data")
    args = parser.parse_args()

    # Check if the target_path is a directory
    if os.path.isdir(args.target_path):
        # If it's a directory, append a default filename
        args.target_path = os.path.join(args.target_path, "aia_12hour_512x512.h5")

    # Ensure the directory for the target file exists
    os.makedirs(os.path.dirname(args.target_path), exist_ok=True)

    process_aia_directory(args.source_path, args.target_path)

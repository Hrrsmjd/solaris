import numpy as np
from torch.utils.data import Dataset

from solaris.utils_data import (
    AIA_INPUT_WAVELENGTHS,
    add_hours,
    load_target_channel,
    load_wavelength_stack,
    read_id_file,
    resolve_id_dir,
)


class CustomDataset_downstream(Dataset):
    def __init__(self, root_dir, data_set="train", id_dir=None):
        self.root_dir = root_dir
        self.data_set = data_set
        self.id_dir = resolve_id_dir(id_dir, data_root=root_dir)
        self.ids = self._get_valid_ids()

    def _get_valid_ids(self):
        """Load valid data IDs from the downstream task ID file."""
        return read_id_file(self.id_dir / f"{self.data_set}_id_1700.txt")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Get input wavelengths and the 1700 channel 12 hours later."""
        current_timestamp = self.ids[idx]
        future_timestamp = add_hours(current_timestamp, 12)

        data = load_wavelength_stack(self.root_dir, current_timestamp, AIA_INPUT_WAVELENGTHS)
        target = load_target_channel(self.root_dir, future_timestamp, "1700")
        return data, target


class CustomDataset_pretrain(Dataset):
    def __init__(self, root_dir, data_set="train", id_dir=None):
        self.root_dir = root_dir
        self.data_set = data_set
        self.id_dir = resolve_id_dir(id_dir, data_root=root_dir)
        self.ids = self._get_valid_ids()

    def _get_valid_ids(self):
        """Load valid data IDs from the pretraining ID file."""
        return read_id_file(self.id_dir / f"{self.data_set}_id.txt")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Get a data sample for pretraining with random target selection."""
        current_timestamp = self.ids[idx]
        future_timestamp = add_hours(current_timestamp, 12)

        data = load_wavelength_stack(self.root_dir, current_timestamp, AIA_INPUT_WAVELENGTHS)
        if np.random.rand() < 0.5:
            target = load_target_channel(self.root_dir, future_timestamp, "1700")
        else:
            target = load_target_channel(self.root_dir, current_timestamp, AIA_INPUT_WAVELENGTHS[-1])

        return data, target

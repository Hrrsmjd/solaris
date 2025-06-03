from torch.utils.data import Dataset
import h5py
import torch
from datetime import datetime
import numpy as np
from solaris.utils_data import *

class CustomDataset_downstream(Dataset):
    def __init__(self, root_dir,data_set = "train"):
        self.root_dir = root_dir
        self.data_set = data_set
        self.ids = self._get_valid_ids()

    def _get_valid_ids(self):
        """Load valid data IDs from the downstream task ID file."""
        valid_ids = []
        with open('CHANGE_PATH' + "train" + '_id_1700.txt', 'r') as file:
            for line in file:
                id_components = list(map(str, line.split()))  
                valid_ids.append(id_components)    
 
        return valid_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Get a data sample with input wavelengths and target 1700 channel 12 hours later."""
        current_timestamp = self.ids[idx]
        future_timestamp = add_hours(current_timestamp, 12)
        
        with h5py.File(self.root_dir + current_timestamp[0] + ".h5", 'r') as file:
            wavelengths = ["0094", "0131", "0171", "0193", "0304", "0335"]
            wavelength_data = []
            for wavelength in wavelengths:
                channel_data = torch.from_numpy(
                    np.array(file[current_timestamp[0]][current_timestamp[1]][current_timestamp[2]][current_timestamp[3]][wavelength], dtype=np.float32)
                )[None, ...]
                wavelength_data.append(channel_data)
            data = torch.cat(wavelength_data, dim=0)
            
        with h5py.File(self.root_dir + future_timestamp[0] + ".h5", 'r') as file:
            target = torch.from_numpy(
                np.array(file[future_timestamp[0]][future_timestamp[1]][future_timestamp[2]][future_timestamp[3]]["1700"], dtype=np.float32)
            )[None, ...]
        
        return data, target

class CustomDataset_pretrain(Dataset):
    def __init__(self, root_dir, data_set="train"):
        self.root_dir = root_dir
        self.data_set = data_set
        self.ids = self._get_valid_ids()
    def _get_valid_ids(self):
        """Load valid data IDs from the pretraining ID file."""
        valid_ids = []
        with open('CHANGE_PATH' + self.data_set + '_id.txt', 'r') as file:
            for line in file:
                id_components = list(map(str, line.split()))  
                valid_ids.append(id_components)    
 
        return valid_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Get a data sample for pretraining with random target selection."""
        current_timestamp = self.ids[idx]
        future_timestamp = add_hours(current_timestamp, 12)
        
        if np.random.rand() < 0.5:
            with h5py.File(self.root_dir + current_timestamp[0] + ".h5", 'r') as file:
                wavelengths = ["0094", "0131", "0171", "0193", "0304", "0335"]
                wavelength_data = []
                for wavelength in wavelengths:
                    channel_data = torch.from_numpy(
                        np.array(file[current_timestamp[0]][current_timestamp[1]][current_timestamp[2]][current_timestamp[3]][wavelength], dtype=np.float32)
                    )[None, ...]
                    wavelength_data.append(channel_data)
                data = torch.cat(wavelength_data, dim=0)
                target = torch.from_numpy(
                    np.array(file[future_timestamp[0]][future_timestamp[1]][future_timestamp[2]][future_timestamp[3]]["1700"], dtype=np.float32)
                )[None, ...]
        else:
            with h5py.File(self.root_dir + current_timestamp[0] + ".h5", 'r') as file:
                wavelengths = ["0094", "0131", "0171", "0193", "0304", "0335"]
                wavelength_data = []
                for wavelength in wavelengths:
                    channel_data = torch.from_numpy(
                        np.array(file[current_timestamp[0]][current_timestamp[1]][current_timestamp[2]][current_timestamp[3]][wavelength], dtype=np.float32)
                    )[None, ...]
                    wavelength_data.append(channel_data)
                data = torch.cat(wavelength_data, dim=0)
                target = torch.from_numpy(
                    np.array(file[current_timestamp[0]][current_timestamp[1]][current_timestamp[2]][current_timestamp[3]][wavelengths[-1]], dtype=np.float32)
                )[None, ...]
        
        return data, target

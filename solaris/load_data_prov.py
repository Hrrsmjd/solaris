from torch.utils.data import Dataset
import h5py
import torch
from datetime import datetime
import numpy as np
from solaris.utils_data import *

class CustomDataset_pretrain(Dataset):
    def __init__(self, root_dir,data_set = "train"):
        self.root_dir = root_dir
        self.data_set = data_set
        self.ids = self._get_valid_ids()
    def _get_valid_ids(self): 
        list_id = []
        with open('CHANGE_PATH'+ self.data_set +'_id.txt', 'r') as file:
            for line in file:
                sublist = list(map(str, line.split()))  
                list_id.append(sublist)    
 
        return list_id 

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        plus_12 = add_hours(i, 12)
        
        with h5py.File(self.root_dir+i[0]+".h5", 'r') as file:
            waves = ["0094", "0131", "0171", "0193", "0304", "0335"]
            data_list = []
            for wave in waves:
                data_list.append(
                    torch.from_numpy(np.array(file[i[0]][i[1]][i[2]][i[3]][wave],dtype=np.float32))[None, ...]
                )
            data = torch.cat(data_list, dim=0)
            target = torch.from_numpy(np.array(file[plus_12[0]][plus_12[1]][plus_12[2]][plus_12[3]]["1700"],dtype=np.float32))[None, ...]
        
        return data, target

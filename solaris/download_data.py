from huggingface_hub import hf_hub_download
import torch
import h5py
import numpy as np
from datetime import datetime,timedelta
from solaris.utils_data import *

path = "CHANGE_PATH"
path_data = path + "/aia_12hour_512x512_"

download_data = False
def download_data(path = path ,year="2023"):
        hf_hub_download(repo_id="hrrsmjd/AIA_12hour_512x512",
                                   repo_type="dataset",filename="aia_12hour_512x512_"+year +".h5",
                                 local_dir=path)
        
if download_data:
    years = ["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022","2023"]
    for year in years:
        download_data(path,year)

def check_data_exists(path = path_data, year="2023"):
    with h5py.File(path+year+".h5", 'r') as f:
        count = 0
        for year_key in f.keys():
            for month_key in f[year_key].keys():
                for day_key in f[year_key][month_key].keys():
                    for hour_key in f[year_key][month_key][day_key].keys():
                        for wavelength_key in f[year_key][month_key][day_key][hour_key].keys():
                            dataset = f[year_key][month_key][day_key][hour_key][wavelength_key]
                            if 'exists' in dataset.attrs and dataset.attrs['exists']:
                                count += 1
        print(f"Total existing data points in {year}: {count}")

def get_valid_ids_for_downstream_task(path = path_data):
    years = ["2019","2020","2021","2022","2023"]
    valid_ids = []
    
    for year in years:
        with h5py.File(path+year+".h5", 'r') as file:
            for year_key in file.keys():
                for month_key in file[year_key].keys():
                    for day_key in file[year_key][month_key].keys():
                        for hour_key in file[year_key][month_key][day_key].keys():
                            i = [year_key, month_key, day_key, hour_key]
                            plus_12 = add_hours(i, 12)
                            
                            try:
                                with h5py.File(path+plus_12[0]+".h5", 'r') as file:
                                    if file[plus_12[0]][plus_12[1]][plus_12[2]][plus_12[3]]["1700"].attrs['exists']:
                                        valid_ids.append(i)
                            except:
                                continue
    
    return valid_ids

def save_the_id_downstram(path = path_data, path_id = 'CHANGE_PATH/train_id_1700.txt'):
    valid_ids = get_valid_ids_for_downstream_task(path)
    
    with open(path_id, 'w') as file:
        for id_list in valid_ids:
            file.write(' '.join(id_list) + '\n')
    
    print(f"Saved {len(valid_ids)} valid IDs to {path_id}")

def save_the_id_pretrain(path = path_data, path_id = 'CHANGE_PATH'):
    years = ["2019","2020","2021","2022","2023"]
    all_ids = []
    
    for year in years:
        with h5py.File(path+year+".h5", 'r') as file:
            for year_key in file.keys():
                for month_key in file[year_key].keys():
                    for day_key in file[year_key][month_key].keys():
                        for hour_key in file[year_key][month_key][day_key].keys():
                            i = [year_key, month_key, day_key, hour_key]
                            all_ids.append(i)
    
    np.random.shuffle(all_ids)
    
    train_split = int(0.8 * len(all_ids))
    val_split = int(0.9 * len(all_ids))
    
    train_ids = all_ids[:train_split]
    val_ids = all_ids[train_split:val_split]
    test_ids = all_ids[val_split:]
    
    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    
    for split, ids in splits.items():
        with open(path_id+"/"+split+"_id.txt", 'w') as file:
            for id_list in ids:
                file.write(' '.join(id_list) + '\n')
        print(f"Saved {len(ids)} {split} IDs")

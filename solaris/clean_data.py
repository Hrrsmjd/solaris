import h5py
import os

def update_exists_attribute(file_path, year, month, day, hour, wavelength):
    """
    Update the 'exists' attribute to False for a specific dataset in HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        year, month, day, hour: Time components for dataset location
        wavelength: Wavelength channel to update
    """
    with h5py.File(file_path, 'r+') as hdf5_file:
        dataset = hdf5_file[year][month][day][hour][wavelength]
        if 'exists' in dataset.attrs and dataset.attrs['exists']:
            dataset.attrs['exists'] = False
            print(f"Successfully changed 'exists' attribute to False for {year}/{month}/{day}/{hour}/{wavelength}")
        else:
            print(f"'exists' attribute is either not present or already False for {year}/{month}/{day}/{hour}/{wavelength}")

data_directory = "CHANGE_PATH"

files_to_process = [
    ('aia_12hour_512x512_2019.h5', [('2019', '01', '13', 'H0000', '0304')]),
    ('aia_12hour_512x512_2021.h5', [('2021', '04', '29', 'H1200', wavelength) for wavelength in ['0094', '0131', '0171', '0193', '0304', '0335']]),
    ('aia_12hour_512x512_2022.h5', [('2022', '02', '04', 'H0000', '0211')]),
]

for filename, dataset_list in files_to_process:
    file_path = os.path.join(data_directory, filename)
    
    if os.path.exists(file_path):
        print(f"Processing file: {filename}")
        for year, month, day, hour, wavelength in dataset_list:
            update_exists_attribute(file_path, year, month, day, hour, wavelength)
    else:
        print(f"File not found: {file_path}")

print("Processing complete.")

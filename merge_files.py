import os
import h5py
import numpy as np

# Folder where rank_*/batch_*.h5 are stored
main_folder = "coefficient_data"     # or wherever you saved them
output_file = "merged_results.h5"

all_coefficients = []
all_parameters = []

# Go through each rank folder
for rank_folder in sorted(os.listdir(main_folder)):
    rank_path = os.path.join(main_folder, rank_folder)
    if not os.path.isdir(rank_path):
        continue

    for batch_file in sorted(os.listdir(rank_path)):
        if batch_file.endswith(".h5"):
            batch_path = os.path.join(rank_path, batch_file)
            with h5py.File(batch_path, 'r') as h5f:
                coeff = h5f["coefficients"][:]
                params = h5f["parameters"][:]
                all_coefficients.append(coeff)
                all_parameters.append(params)

# Stack everything together
all_coefficients = np.vstack(all_coefficients)
all_parameters = np.vstack(all_parameters)

# Write final merged file
with h5py.File(os.path.join(main_folder, output_file), 'w') as h5f:
    h5f.create_dataset("coefficients", data=all_coefficients)
    h5f.create_dataset("parameters", data=all_parameters)

print(f"Merged file written to {os.path.join(main_folder, output_file)}")
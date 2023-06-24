import os
import numpy as np
import pandas as pd
import h5py

NUM_GPUS = 10
MISSING_HEADER = True

def roundup(x, base=50):
    return x if x % base == 0 else x + base - x % base

base_csv = pd.read_csv('./expressions.csv', header=(None if MISSING_HEADER else 0))
base_h5f = h5py.File('./parameters.h5', 'a')

if MISSING_HEADER:
    base_csv.columns = ['Unnamed: 0', 'expr', 'index', 'best_mse_loss', 'best_scaled_mse_loss']

assert base_h5f['nn_parameters'].shape[0] == roundup(base_csv.shape[0], 50)
assert base_h5f['counter'][:].item() == base_csv.shape[0]

for i in range(1, NUM_GPUS):
    if not os.path.exists(f"./expressions_{i:02d}.csv"):
        continue
    if not os.path.exists(f"./parameters_{i:02d}.h5"):
        continue

    update_csv = pd.read_csv(f"./expressions_{i:02d}.csv", header=(None if MISSING_HEADER else 0))
    update_h5f = h5py.File(f"./parameters_{i:02d}.h5", 'r')
    
    if MISSING_HEADER:
        update_csv.columns = ['Unnamed: 0', 'expr', 'index', 'best_mse_loss', 'best_scaled_mse_loss']

    assert update_h5f['nn_parameters'].shape[0] == roundup(update_csv.shape[0], 50)
    assert update_h5f['counter'][:].item() == update_csv.shape[0]

    assert len(update_csv.columns) == len(base_csv.columns)
    update_csv.columns = base_csv.columns
    update_csv['index'] += base_csv.shape[0]

    base_h5f['nn_parameters'].resize(
        (roundup(base_csv.shape[0] + update_csv.shape[0], 50), base_h5f['nn_parameters'].shape[1])
    )
    update_parameters = update_h5f['nn_parameters'][:update_csv.shape[0], :]
    base_h5f['nn_parameters'][base_csv.shape[0]:(base_csv.shape[0] + update_csv.shape[0]), :] = update_parameters
    base_h5f['counter'][:] = base_csv.shape[0] + update_csv.shape[0]
    update_h5f.close()

    base_csv = pd.concat([base_csv, update_csv], axis=0, ignore_index=True)

print(base_h5f['counter'][:].item())
print(base_h5f['nn_parameters'].shape)
print(base_csv.shape)
base_h5f.close()
base_csv = base_csv.drop('Unnamed: 0', axis=1)
base_csv.to_csv("./expressions.csv")
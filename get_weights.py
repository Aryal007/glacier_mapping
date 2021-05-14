import numpy as np
import os, pathlib 

data_dir = pathlib.Path("/datadrive/DynamicEarthNet/processed/train")

files = [x for x in os.listdir(data_dir) if x.startswith("mask")]

arr = np.zeros((len(files), 7))

for i, f in enumerate(files):
    _arr = np.load(data_dir / f)
    arr[i] = np.sum(_arr, axis=(0,1)) / (512*512)

distribution = np.sum(arr, axis=0)
distribution = distribution / np.sum(distribution)
print(f"Samples distribution: {np.round(distribution, 5)}")
weights = np.prod(distribution)/distribution
weights = weights*1e8
print(f"Weights: {np.round(weights, 5)}")
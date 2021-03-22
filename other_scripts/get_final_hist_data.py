import numpy as np 
import yaml, json, pathlib, os

def get_iou(y_tp, y_t, y_fn):
    return y_tp/(y_t + y_fn)

data_dir = pathlib.Path("/mnt/datadrive/noaa/hist_arr/")
save_location = pathlib.Path("/mnt/datadrive/noaa/hist_arr/hists")

files = [x for x in os.listdir(data_dir) if x.endswith("npy")] 

for f in files:
    arr = np.round(np.load(data_dir / f), 2)
    if "ndswi" in f or "ndwi" in f:
        bins = np.linspace(-1, 1, 101)
        np.save(save_location / "wi_bins", bins)
    else:
        bins = np.linspace(0, 1, 101)
        np.save(save_location / "other_bins", bins)
    hist, bins = np.histogram(arr, bins = bins)
    np.save(save_location / f, hist)
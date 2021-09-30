import numpy as np 
import yaml, json, pathlib, os
import matplotlib.pyplot as plt

def get_iou(y_tp, y_fp, y_fn):
    return y_tp/(y_tp + y_fp + y_fn)

data_dir = pathlib.Path("/mnt/datadrive/noaa/hist_arr/hists")

files = [x for x in os.listdir(data_dir) if x.endswith("npy") and "water" in x]

names = {'ndwi': "NDWI",
        'ndswi': "NDSWI",
        'xgboost': "XGBoost",
        'unet': "UNet",
        'rf': "Random Forest",
        'unet4': 'UNet Original'}

for f in files:
    bins = np.load(data_dir / "other_bins.npy")
    xlower = 0
    if "ndswi" in f or "ndwi" in f:
        bins = np.load(data_dir / "wi_bins.npy")
        xlower = -1
    width = np.diff(bins)
    bins = bins[:-1]
    land_ious = np.zeros_like(bins)
    water_ious = np.zeros_like(bins)
    land_f = f.replace("water", "land")
    water = np.load(data_dir / f)
    land = np.load(data_dir / land_f)
    for i, threshold in enumerate(bins):
        tp_water = np.sum(water[bins >= threshold])
        fp_water = np.sum(land[bins >= threshold])
        fn_water = np.sum(water[bins < threshold])
        water_ious[i] = get_iou(tp_water, fp_water, fn_water)*100
        
        tp_land = np.sum(land[bins < threshold])
        fp_land = fn_water
        fn_land = fp_water
        land_ious[i] = get_iou(tp_land, fp_land, fn_land)*100

    plt.bar(bins, water*100/np.sum(water), alpha=0.3, label="Water", width=width, edgecolor="black", align="edge")
    plt.bar(bins, land*100/np.sum(land), alpha=0.3, label="Land", width=width, edgecolor="black", align="edge")
    plt.plot(bins, water_ious, label="Water IOU")
    plt.plot(bins, land_ious, label="Land IOU")
    plt.plot(bins[np.argmax(water_ious)], np.max(water_ious), color = "red", marker="*")
    plt.text(bins[np.argmax(water_ious)], np.max(water_ious)-5, str(np.round(bins[np.argmax(water_ious)], 2))+", "+str(np.round(np.max(water_ious), 2)), color='b', fontsize=12)
    #plt.text(bins[np.argmax(land_ious)], np.max(land_ious)-0.05, str(np.round(bins[np.argmax(land_ious)], 2))+", "+str(np.round(np.max(land_ious), 2)), color='orange', fontsize=12)
    plt.legend(loc='best')
    plt.title(names[f.split("_")[0]])
    plt.ylabel('Percentage')
    plt.xlabel('Intensity')
    plt.xlim(xlower, 1)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(f.split("_")[0]+".png")
    plt.cla()
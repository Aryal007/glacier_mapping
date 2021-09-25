from pathlib import Path
import numpy as np
import glob, os, shutil, yaml
from tifffile import imsave, imread
import rasterio, pdb
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import median

val_ids = ['hbe', 'tnp', 'wvy', 'ayt']
min_values = np.array([-33.510303, -39.171803, -182.45174])
max_values = np.array([7.2160087, 2.8161404, 40.3697])

def get_image(vv, vh, smooth=False):
    if smooth:
        vv = median(vv, disk(2))
        vh = median(vh, disk(2))
    blue = np.nan_to_num(vv / vh)
    img = np.concatenate((np.clip(vv[:,:,None], min_values[0], max_values[0]), 
                        np.clip(vh[:,:,None], min_values[1], max_values[1]), 
                        np.clip(blue[:,:,None], min_values[2], max_values[2])), 
                        axis=2)
    return img

def add_rsi(img):
    out = np.zeros((img.shape[0], img.shape[1], 6))
    out[:,:,:3] = img
    vv = img[:,:,0]
    vh = img[:,:,1]
    blue = img[:,:,2]
    nprb = np.clip(np.nan_to_num((vv - vh) / (vv + vh)), -1, 1)
    bi = np.sqrt(np.square(vv)+np.square(vh)+np.square(blue))/3
    ndwi = np.clip(np.nan_to_num((blue - vv - vh) / (blue + vv + vh)), -1, 1)
    out[:,:,3] = nprb
    out[:,:,4] = bi
    out[:,:,5] = ndwi
    return out

def add_supplementary(img, t):
    n_channels = img.shape[2]
    out = np.zeros((img.shape[0], img.shape[1], n_channels+7))
    out[:,:,:n_channels] = img
    change = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","jrc-gsw-change")))), -1000, 1000)
    extent = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","jrc-gsw-extent")))), -1000, 1000)
    occurrence = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","jrc-gsw-occurrence")))), -1000, 1000)
    recurrence = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","jrc-gsw-recurrence")))), -1000, 1000)
    seasonality = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","jrc-gsw-seasonality")))), -1000, 1000)
    transitions = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","jrc-gsw-transitions")))), -1000, 1000)
    nasadem = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","nasadem")))), -1000, 1000)
    out[:,:,n_channels] = change
    out[:,:,n_channels+1] = extent
    out[:,:,n_channels+2] = occurrence
    out[:,:,n_channels+3] = recurrence
    out[:,:,n_channels+4] = seasonality
    out[:,:,n_channels+5] = transitions
    out[:,:,n_channels+6] = nasadem
    return out

def remove_outliers(img):
    mean = np.mean(img, axis=(0,1))
    std = np.std(img, axis=(0,1))
    img = np.clip(img, mean-3*std, mean+3*std)
    return img

def min_max(img):
    img = (img - min_values) / (max_values - min_values)
    img[:,:,2] = img[:,:,2]/1.5
    return img

if __name__ == "__main__":
    data_dir = Path("./data")
    features_dir = data_dir / "train_features"
    labels_dir = data_dir / "train_labels"
    if not os.path.exists(data_dir / "processed"):
        os.mkdir(data_dir / "processed")
    if not os.path.exists(data_dir / "processed" / "train"):
        os.mkdir(data_dir / "processed" / "train")
    if not os.path.exists(data_dir / "processed" / "val"):
        os.mkdir(data_dir / "processed" / "val")
    tiff_files = sorted(features_dir.glob("*_vv.tif"))
    label_files = sorted(labels_dir.glob("*.tif"))

    means, stds = [], []
    val_count, train_count = 0, 0
    background, floodwater = 0, 0

    for l, t in zip(label_files, tiff_files):
        vv_filename = t
        vh_filename = Path(str(t).replace("vv","vh"))
        flag = 0
        for val_id in val_ids:
            if val_id in str(l):
                flag = 1

        lab = imread(l)[:,:,None]
        vv = imread(vv_filename)
        mask = np.zeros_like(np.squeeze(vv))
        with rasterio.open(t) as img:
            vv_numpy_mask = img.read(1, masked=True)
        lab[vv_numpy_mask.mask] = 255
        vh = imread(vh_filename)
        with rasterio.open(vh_filename) as img:
            vh_numpy_mask = img.read(1, masked=True)
        lab[vh_numpy_mask.mask] = 255

        if flag:
            img = get_image(vv, vh, smooth=True)
        else:
            img = get_image(vv, vh, smooth=True)

        img = remove_outliers(img)
        img = min_max(img)
        img = add_rsi(img)
        img = add_supplementary(img, t)
        if flag:
            val_count += 1
            lab_out_fname = str(l).replace(".tif","").replace("train_labels/","processed/val/mask_")
            out_fname = str(t).replace("_vv.tif","").replace("train_features/","processed/val/tiff_")
        else:
            train_count += 1
            lab_out_fname = str(l).replace(".tif","").replace("train_labels/","processed/train/mask_")
            out_fname = str(t).replace("_vv.tif","").replace("train_features/","processed/train/tiff_")
            _temp = np.squeeze(lab)
            background += np.sum(_temp == 0)
            floodwater += np.sum(_temp == 1)
            mean = np.mean(img, axis=(0,1))
            std = np.std(img, axis=(0,1))
            means.append(mean)
            stds.append(std)
        np.save(lab_out_fname, lab)
        np.save(out_fname, img)

    means = np.asarray(means)
    stds = np.asarray(stds)
    print(f"\nMean: {np.mean(means, axis=0)}")
    print(f"\nStandard Deviation: {np.mean(stds, axis=0)}")
    print(f"\nValidation Samples: {val_count}, Training Samples: {train_count}, Validation %: {(val_count/(val_count+train_count))*100:.2f}")
    print(f"\nBackground pixels: {background}, Flood pixels: {floodwater}, Ratio: {floodwater/background:.3f}")
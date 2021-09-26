from pathlib import Path
import numpy as np
import glob, os, shutil, yaml
from tifffile import imsave, imread
import rasterio, pdb
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import median

val_ids = ['qxb']
min_values = np.array([-33.510303, -39.171803, -182.45174])
max_values = np.array([7.2160087, 2.8161404, 40.3697])
supplementary_min_values = np.array([0, 0, 0, 0, 0, 0, 0])
supplementary_max_values = np.array([255, 255, 255, 255, 255, 1, 1000])

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
    change = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","jrc-gsw-change")))), supplementary_min_values[0], supplementary_max_values[0])[:,:,None]
    extent = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","jrc-gsw-extent")))), supplementary_min_values[1], supplementary_max_values[1])[:,:,None]
    occurrence = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","jrc-gsw-occurrence")))), supplementary_min_values[2], supplementary_max_values[2])[:,:,None]
    recurrence = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","jrc-gsw-recurrence")))), supplementary_min_values[3], supplementary_max_values[3])[:,:,None]
    seasonality = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","jrc-gsw-seasonality")))), supplementary_min_values[4], supplementary_max_values[4])[:,:,None]
    transitions = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","jrc-gsw-transitions")))), supplementary_min_values[5], supplementary_max_values[5])[:,:,None]
    nasadem = np.clip(np.nan_to_num(imread(Path(str(t).replace("vv","nasadem")))), supplementary_min_values[6], supplementary_max_values[6])[:,:,None]
    supplementary = np.concatenate((change, extent, occurrence, recurrence, seasonality, transitions, nasadem), axis=2)
    supplementary = (supplementary - supplementary_min_values) / (supplementary_max_values - supplementary_min_values)
    out[:,:,n_channels:] = supplementary
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
        np.save(lab_out_fname, lab)
        np.save(out_fname, img)

    print(f"\nValidation Samples: {val_count}, Training Samples: {train_count}, Validation %: {(val_count/(val_count+train_count))*100:.2f}")
    print(f"\nBackground pixels: {background}, Flood pixels: {floodwater}, Ratio: {floodwater/background:.3f}")
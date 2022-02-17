#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:27:56 2021

@author: mibook
""" 
import os, yaml
from pathlib import Path
import numpy as np
from addict import Dict
import segmentation.data.slice as fn
import warnings, pdb
import pandas as pd
warnings.filterwarnings("ignore")

conf = Dict(yaml.safe_load(open('./conf/slice_and_preprocess.yaml')))

df = pd.read_csv(Path(conf.image_dir) / 'metadata.csv')
saved_df = pd.DataFrame(columns=["Landsat ID", "Image", "Slice", 
                                "Background", "Clean Ice", "Debris", "Masked",
                                "Background Percentage", "Clean Ice Percentage", "Debris Percentage", "Masked Percentage", 
                                "split"])
train_df = df[df.split == "train"]
val_ids = [
            '133041','133040','134040','135040',
            '138041','147037','149034','150036',
            '152034','146038','147036','144039'
        ]
val_filenames = sorted([x+'.tif' for x in train_df.image_id if x.split("_")[1] in val_ids])
train_filenames = sorted([x+'.tif' for x in train_df.image_id if x.split("_")[1] not in val_ids])
val_filenames = [Path(conf.image_dir) / x for x in val_filenames]
train_filenames = [Path(conf.image_dir) / x for x in train_filenames]
label_filename = Path(conf.labels_dir) / "train.shp"
roi_filename = Path(conf.labels_dir) / "train_roi.shp"
shp = fn.read_shp(label_filename)
roi_shp = fn.read_shp(roi_filename)

fn.remove_and_create(conf.out_dir)

means, stds, mins, maxs = [], [], [], []
savepath = Path(conf["out_dir"]) / 'train'
fn.remove_and_create(savepath)
for i, train_filename in enumerate(train_filenames):
    print(f"Filename: {train_filename.name}")
    tiff = fn.read_tiff(train_filename)
    mask = fn.get_mask(tiff, shp)
    roi_mask = fn.get_mask(tiff, roi_shp, column="CONTINENT")
    mean, std, _min, _max, saved_df = fn.save_slices(train_filename.name, i, tiff, mask, roi_mask, savepath, saved_df, **conf)
    means.append(mean) 
    stds.append(std)
    mins.append(_min) 
    maxs.append(_max)

print("Saving training slices completed!!!")
means = np.mean(np.asarray(means), axis=0)
stds = np.mean(np.asarray(stds), axis=0)
mins = np.min(np.asarray(mins), axis=0)
maxs = np.mean(np.asarray(maxs), axis=0)

np.save(conf.out_dir+"normalize_train", np.asarray((means, stds, mins, maxs)))

means, stds, mins, maxs = [], [], [], []
savepath = Path(conf["out_dir"]) / 'val'
fn.remove_and_create(savepath)
for i, val_filename in enumerate(val_filenames[10:]):
    print(f"Filename: {val_filename.name}")
    tiff = fn.read_tiff(val_filename)
    mask = fn.get_mask(tiff, shp)
    roi_mask = fn.get_mask(tiff, roi_shp, column="CONTINENT")
    mean, std, _min, _max, saved_df = fn.save_slices(val_filename.name, i, tiff, mask, roi_mask, savepath, saved_df, **conf)
    means.append(mean) 
    stds.append(std)
    mins.append(_min) 
    maxs.append(_max)

print("Saving validation slices completed!!!")
means = np.mean(np.asarray(means), axis=0)
stds = np.mean(np.asarray(stds), axis=0)
mins = np.min(np.asarray(mins), axis=0)
maxs = np.mean(np.asarray(maxs), axis=0)

np.save(conf.out_dir+"normalize_val", np.asarray((means, stds, mins, maxs)))
saved_df.to_csv(conf.out_dir+"slice_meta.csv", encoding='utf-8', index=False)

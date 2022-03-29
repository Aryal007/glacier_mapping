#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:27:56 2021

@author: Aryal007
"""
import os
import yaml
from pathlib import Path
import numpy as np
from addict import Dict
import segmentation.data.slice as fn
import warnings
import pdb
import pandas as pd
warnings.filterwarnings("ignore")

conf = Dict(yaml.safe_load(open('./conf/slice_and_preprocess.yaml')))

df = pd.read_csv(Path(conf.image_dir) / 'metadata.csv')
saved_df = pd.DataFrame(columns=["Landsat ID", "Image", "Slice", "Background",
                                "Clean Ice", "Debris", "Masked", "Background Percentage",
                                "Clean Ice Percentage", "Debris Percentage",
                                "Masked Percentage", "split"])
train_df = df[df.split == "train"]
test_df = df[df.split == "test"]
val_ids = [
    '133041', '133040', '134040', '137041',
    '138041', '144039', '145039', '146038', 
    '147037', '148035', '150036', '152034', 
]
train_filenames = sorted(list(
    set([x + '.tif' for x in train_df.image_id if x.split("_")[1] not in val_ids])))
val_filenames = sorted(
    list(set([x + '.tif' for x in train_df.image_id if x.split("_")[1] in val_ids])))
test_filenames = sorted([x+'.tif' for x in test_df.image_id])

splits = {
    'train': {'filename': train_filenames,
              'shp': fn.read_shp(Path(conf.labels_dir) / "train.shp"),
              'roi': fn.read_shp(Path(conf.labels_dir) / "train_roi.shp")},
    'val' : {'filename': val_filenames,
              'shp': fn.read_shp(Path(conf.labels_dir) / "train.shp"),
              'roi': fn.read_shp(Path(conf.labels_dir) / "train_roi.shp")},
    'test' : {'filename': test_filenames,
              'shp': fn.read_shp(Path(conf.labels_dir) / "test.shp"),
              'roi': fn.read_shp(Path(conf.labels_dir) / "test_roi.shp")},
}

fn.remove_and_create(conf.out_dir)

for split, meta in splits.items():
    means, stds, mins, maxs = [], [], [], []
    savepath = Path(conf["out_dir"]) / split
    fn.remove_and_create(savepath)
    for i, filename in enumerate(meta['filename']):
        filename = Path(conf.image_dir) / filename
        print(f"Filename: {filename.name}")
        tiff = fn.read_tiff(filename)
        mask = fn.get_mask(tiff, meta['shp'])
        roi_mask = fn.get_mask(tiff, meta['roi'], column="CONTINENT")
        mean, std, _min, _max, saved_df = fn.save_slices(
        filename.name, i, tiff, mask, roi_mask, savepath, saved_df, **conf)
        means.append(mean)
        stds.append(std)
        mins.append(_min)
        maxs.append(_max)
    print(f"Saving {split} slices completed!!!")
    means = np.mean(np.asarray(means), axis=0)
    stds = np.mean(np.asarray(stds), axis=0)
    mins = np.min(np.asarray(mins), axis=0)
    maxs = np.max(np.asarray(maxs), axis=0)
    np.save(conf.out_dir + f"normalize_{split}", np.asarray((means, stds, mins, maxs)))

saved_df.to_csv(conf.out_dir + "slice_meta.csv", encoding='utf-8', index=False)

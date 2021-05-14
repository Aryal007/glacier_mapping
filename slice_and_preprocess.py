#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:27:56 2021

@author: mibook
""" 
import os, yaml, pathlib
import numpy as np
from addict import Dict
import coastal_mapping.data.slice as fn
import warnings, pdb
from PIL import Image
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
np.random.seed(7)

conf = Dict(yaml.safe_load(open('./conf/slice_and_preprocess.yaml')))
labels_dir = pathlib.Path(conf.labels_dir)
image_dir = pathlib.Path(conf.image_dir)
out_dir = pathlib.Path(conf.out_dir)
split_dir = pathlib.Path(conf.split_dir)
fn.remove_and_create(out_dir)

#vals = ['21S_2832_4366_13', '33N_4426_3835_13', '43N_5863_3800_13', '48N_6466_3380_13']
vals = [0,3,7]

prev_year, prev_month = '2018', '00'

means, stds = [], []
counts = []

for zone in sorted(os.listdir(image_dir)):
    if os.path.isdir(image_dir / zone):
        for cubeid in sorted(os.listdir(image_dir / zone)):
            #print("New cubeid")
            if os.path.isdir(image_dir / zone / cubeid):
                image_num = 0
                for tiff in sorted(os.listdir(image_dir / zone / cubeid / 'L3H-SR' )):
                    if tiff.endswith('01.tif'):
                        tiff_filename = image_dir / zone / cubeid / 'L3H-SR' / tiff
                        year = tiff.split("-")[0]
                        month = tiff.split("-")[1]
                        labels_dir1 = zone + "_" + cubeid
                        _filename = zone + "_" + cubeid + "-" + year + '-' + month + '-' + prev_year + '-' + prev_month
                        label_filename = labels_dir / labels_dir1 / (_filename + ".png")
                        print(f"Filename: {zone}_{cubeid}-{year}-{month}, Imagenum: {image_num}")
                        if os.path.exists(label_filename):
                            _tiff = fn.read_tiff(tiff_filename)
                            _mask = np.array(Image.open(label_filename))
                            if image_num > 0:
                                prev_image[_mask != 0] = 0
                                print(f"Original, Augmented = {np.sum(_mask)}, {np.sum(_mask + prev_image)}")
                                _mask = _mask + prev_image
                            randint = np.random.random_integers(10)
                            if randint in vals:
                                _conf = conf.copy()
                                _conf["out_dir"] = conf["out_dir"] + "val/"
                                print("\t\tSaved Validation data")
                                mean, std = fn.save_slices(_filename, _tiff, _mask, **_conf)
                            else:
                                _conf = conf.copy()
                                _conf["out_dir"] = conf["out_dir"] + "train/"
                                mean, std = fn.save_slices(_filename, _tiff, _mask, **_conf)
                                means.append(mean) 
                                stds.append(std)
                                print("\t\tSaved Training data")
                            image_num += 1
                            prev_image = _mask
                        else:
                            print(f"Discarding {zone}_{cubeid}-{year}-{month}")
                        prev_year, prev_month = str(year), str(month)

print("Saving slices completed!!!")
means = np.mean(np.asarray(means), axis=0)
stds = np.mean(np.asarray(stds), axis=0)

np.save(conf.out_dir+"normalize", np.asarray((means, stds)))
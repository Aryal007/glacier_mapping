#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:27:56 2021

@author: mibook
""" 
import os, yaml
import numpy
from addict import Dict
import coastal_mapping.data.slice as fn

conf = Dict(yaml.safe_load(open('./conf/slice.yaml')))

label_filenames = [x for x in os.listdir(conf.labels_dir) if x.endswith('.shp')]
tiff_filenames = [x.replace('shp','TIF') for x in label_filenames]

fn.remove_and_create(conf.out_dir)

for i, (label_filename, tiff_filename) in enumerate(zip(label_filenames, tiff_filenames)):
    print(label_filename)
    shp = fn.read_shp(conf.labels_dir+label_filename)
    tiff = fn.read_tiff(conf.image_dir+tiff_filename)
    
    mask = fn.get_mask(tiff, shp)
    
    fn.save_slices(i, tiff, mask, **conf)

print("Saving slices completed!!!")

if conf.train_split+conf.val_split+conf.test_split != 1:
    raise ValueError("Sum of train, test, val split should be 1!")

fn.train_test_shuffle(conf.out_dir, conf.train_split, conf.val_split, conf.test_split)

print("Shuffle complete...")
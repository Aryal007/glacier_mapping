#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 21:52:29 2021

@author: mibook
"""
import numpy as np
from coastal_mapping.data.slice import remove_and_create
import yaml, json, pathlib, os
from addict import Dict

def getXy(tiff, mask, n_sample):
    np_tiff = np.load(tiff)
    np_mask = np.load(mask)
    np_tiff[np_tiff == -32767] = np.nan
    nonnan_mask = ~np.isnan(np.mean(np_tiff[:,:,:-2], axis=2))
    land_mask = np_mask[:,:,0] == 1
    land_mask = land_mask * nonnan_mask
    water_mask = np_mask[:,:,1] == 1
    water_mask = water_mask * nonnan_mask

    water_tiff = np_tiff[water_mask]
    land_tiff = np_tiff[land_mask]

    if np.min((len(water_tiff), n_sample)) != 0:
        random_index = np.random.permutation(len(water_tiff))[:n_sample]
        X = water_tiff[random_index,:]
        y = np.ones(np.min((len(water_tiff), n_sample)))
    if np.min((len(land_tiff), n_sample)) != 0:
        random_index = np.random.permutation(len(land_tiff))[:n_sample]
        try:
            X = np.vstack((X, land_tiff[random_index,:]))
            y = np.hstack((y, np.zeros(np.min((len(land_tiff), n_sample)))))
        except Exception as e:
            X = land_tiff[random_index,:]
            y = np.zeros(np.min((len(land_tiff), n_sample)))

    return X, y

if __name__ == "__main__":
    conf = Dict(yaml.safe_load(open('./conf/ml_prepareXY.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    out_dir = pathlib.Path(conf.out_dir)
    
    remove_and_create(out_dir)
    
    print("Preparing data")
    
    for d in conf.data:
        path = data_dir / d
        tiff_filenames = [x for x in os.listdir(path) if x[:4] == "tiff"]

        for i, tiff_filename in enumerate(tiff_filenames):
            mask_filename = tiff_filename.replace('tiff', 'mask')
            _X, _y = getXy(path/tiff_filename, path/mask_filename, conf.per_class_pixels_per_slice)
            try:
                X = np.vstack((X, _X))
                y = np.hstack((y, _y))
            except:
                X, y = _X, _y
            if (i+1)%conf.save_every_files == 0:
                # save here
                np.save(out_dir / ("X_" + d + "_" + str(i+1)), X)
                np.save(out_dir / ("y_" + d + "_" + str(i+1)), y)
                print(f"Saved {i+1} {d}")
                print(f"Number of samples: {len(y)}")
                del(X)
                del(y)

        np.save(out_dir / ("X_" + d + "_" + str(i+1)), X)
        np.save(out_dir / ("y_" + d + "_" + str(i+1)), y)

        print(f"Completed {d}")
        del(X)
        del(y)
    
    print("Merging arrays")
    
    for d in conf.data:
        X_filenames = [x for x in os.listdir(out_dir) if (d in x and x[0] == "X")]
        for X_filename in X_filenames:
            y_filename = X_filename.replace("X","y")
            _X = np.load(out_dir / X_filename)
            _y = np.load(out_dir / y_filename)
            try:
                X = np.vstack((X, _X))
                y = np.hstack((y, _y))
            except:
                X = _X
                y = _y
        
        for x in X_filenames:
            os.remove(out_dir/x)
            os.remove(out_dir/x.replace("X", "y"))
        
        print(f"{d} samples: {len(y)}")

        print(f"Shuffling {d}")
        randidx = np.random.permutation(len(y))
        np.save(out_dir / ("X_" + d), X[randidx, :])
        np.save(out_dir / ("y_" + d), y[randidx])    
        del(X)
        del(y)
    
    print("Data preparation complete...")
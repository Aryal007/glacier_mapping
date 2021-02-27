#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 2:57:34 2021

@author: mibook
"""
from coastal_mapping.data.data import fetch_loaders
from coastal_mapping.model.frame import Framework
from coastal_mapping.data.slice import add_index
import coastal_mapping.model.functions as fn

import yaml, pathlib
from addict import Dict
import rasterio
import numpy as np
import warnings
import pdb
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    conf = Dict(yaml.safe_load(open('./conf/predict.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    tif_path = data_dir / "images" / conf.filename
    model_path = data_dir / "runs" / conf.run_name / "models" / "model_final.pt"

    tiff = rasterio.open(tif_path)
    tiff_np = tiff.read()

    loss_fn = fn.get_loss(conf.model_opts.args.outchannels)    
        
    frame = Framework(
        loss_fn = loss_fn,
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts
    )

    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    frame.load_state_dict(state_dict)
    
    if tiff_np.dtype == "uint16":
        tiff_np = tiff_np/255

    tiff_np = tiff_np.transpose(1, 2, 0)
    if conf["add_ndvi"]:
        tiff_np = add_index(tiff_np, index1 = 3, index2 = 0, comment = "ndvi")
    if conf["add_ndwi"]:
        tiff_np = add_index(tiff_np, index1 = 1, index2 = 3, comment = "ndwi")
    if conf["add_ndswi"]:
        tiff_np = add_index(tiff_np, index1 = 3, index2 = 2, comment = "ndswi")
    
    tiff_np = tiff_np[:,:,conf.use_channels]
    arr = np.load(data_dir / "processed" / "normalize.npy")
    mean, std = arr[0][conf.use_channels], arr[1][conf.use_channels]
    orig_image = tiff_np
    tiff_np = (tiff_np - mean) / std

    x = np.expand_dims(tiff_np, axis=0)
    y = np.zeros((x.shape[1], x.shape[2]))

    x = torch.from_numpy(x).float()

    for row in range(0, x.shape[1], conf.window_size[0]):
        for column in range(0, x.shape[2], conf.window_size[1]):
            current_slice = x[:, row:row+conf["window_size"][0], column:column+conf["window_size"][1], :]
            if current_slice.shape[1] != conf.window_size[0] or current_slice.shape[2] != conf.window_size[1]:
                temp = np.zeros((1, conf.window_size[0], conf.window_size[1], x.shape[3]))
                temp[:, :current_slice.shape[1], :current_slice.shape[2], :] =  current_slice
                current_slice = torch.from_numpy(temp).float()
            mask = current_slice.squeeze().sum(dim=2) == 0
            prediction = frame.infer(current_slice)
            prediction = np.argmax(np.asarray(prediction.cpu()).squeeze(), axis=2)+1
            prediction[mask] = 0
            endrow_dest = row+conf.window_size[0]
            endrow_source = conf.window_size[0]
            endcolumn_dest = column+conf.window_size[0]
            endcolumn_source = conf.window_size[1]
            if endrow_dest > y.shape[0]:
                endrow_source = y.shape[0] - row
                endrow_dest = y.shape[0]
            if endcolumn_dest > y.shape[1]:
                endcolumn_source = y.shape[1] - column
                endcolumn_dest = y.shape[1]
            try:
                y[row:endrow_dest, column:endcolumn_dest] = prediction[0:endrow_source, 0:endcolumn_source]
            except:
                print("Something wrong with indexing!")
    
    plt.imsave("./image.png", orig_image[:,:,:3].clip(0,1))
    plt.imsave("./mask.png", y) 
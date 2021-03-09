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

import yaml, pathlib, pickle
from addict import Dict
import rasterio
import numpy as np
import warnings
import pdb
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

warnings.filterwarnings("ignore")

font = {'size'   : 20}

matplotlib.rc('font', **font)

if __name__ == "__main__":
    conf = Dict(yaml.safe_load(open('./conf/unet_predict.yaml')))
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

    filename = conf.filename.split(".")[0]

    arr = np.load(data_dir / "processed" / "normalize.npy")
    mean, std = arr[0][conf.use_channels], arr[1][conf.use_channels]
    orig_image = tiff_np
    tiff_np = (tiff_np - mean) / std

    x = np.expand_dims(tiff_np, axis=0)
    y = np.zeros((x.shape[1], x.shape[2]))
    y_rf = np.zeros((x.shape[1], x.shape[2]))
    y_xgboost = np.zeros((x.shape[1], x.shape[2]))

    rf_model = pickle.load(open("/mnt/datadrive/noaa/ml_data/random_forest/estimator.sav", 'rb'))
    xgboost_model = pickle.load(open("/mnt/datadrive/noaa/ml_data/xgboost/estimator.sav", 'rb'))

    x = torch.from_numpy(x).float()

    for row in range(0, x.shape[1], conf.window_size[0]):
        for column in range(0, x.shape[2], conf.window_size[1]):
            current_slice = x[:, row:row+conf["window_size"][0], column:column+conf["window_size"][1], :]
            if current_slice.shape[1] != conf.window_size[0] or current_slice.shape[2] != conf.window_size[1]:
                temp = np.zeros((1, conf.window_size[0], conf.window_size[1], x.shape[3]))
                temp[:, :current_slice.shape[1], :current_slice.shape[2], :] =  current_slice
                current_slice = torch.from_numpy(temp).float()
            mask = current_slice.squeeze()[:,:,:3].sum(dim=2) == 0
            prediction = frame.infer(current_slice)
            prediction = torch.nn.Softmax(3)(prediction)
            prediction = np.asarray(prediction.cpu()).squeeze()[:,:,1]
            prediction[mask] = 0

            _current_slice = np.asarray(current_slice.cpu()).squeeze()
            _current_slice = _current_slice.reshape((_current_slice.shape[0]*_current_slice.shape[1]), _current_slice.shape[2])
            prediction_rf = rf_model.predict_proba(_current_slice)
            prediction_xgboost = xgboost_model.predict_proba(_current_slice)
            prediction_rf = prediction_rf.reshape(current_slice.shape[1], current_slice.shape[2], prediction_rf.shape[1])
            prediction_xgboost = prediction_xgboost.reshape(current_slice.shape[1], current_slice.shape[2], prediction_xgboost.shape[1])
            prediction_rf = prediction_rf[:,:,1]
            prediction_xgboost = prediction_xgboost[:,:,1]

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
                y_rf[row:endrow_dest, column:endcolumn_dest] = prediction_rf[0:endrow_source, 0:endcolumn_source]
                y_xgboost[row:endrow_dest, column:endcolumn_dest] = prediction_xgboost[0:endrow_source, 0:endcolumn_source]
            except Exception as e:
                print("Something wrong with indexing!")
            
    fig, plots = plt.subplots(nrows = 2, ncols=3, figsize=(20, 20))
    images = [orig_image[:,:,:3], (orig_image[:,:,5]+1/2), (orig_image[:,:,6]+1)/2, (y-0.1)/0.8, y_rf, y_xgboost]
    titles = ["Original Image", "NDWI", "NDSWI", "U-Net Prediction", "Random Forest", "XGBoost"]

    for i, graphs in enumerate(plots.flat):
        im = graphs.imshow(images[i], vmin=0, vmax=1)
        graphs.set_title(titles[i], fontsize=20)
        graphs.axis('off')

    fig.suptitle("Multi Approach Water Intensity Masks", fontsize=28)
    plt.colorbar(im, ax=plots.ravel().tolist(), label="Water Intensity", orientation="horizontal")
    plt.savefig(filename+".png")
    plt.close(fig)

    fig, plots = plt.subplots(ncols=5, figsize = (20,4), sharey=False, tight_layout=True)
    hists = [orig_image[:,:,5].flatten(), orig_image[:,:,6].flatten(), y.flatten(), y_rf.flatten(), y_xgboost.flatten()]
    titles = ["NDWI", "NDSWI", "U-Net", "Random Forest", "XGBoost"]
    for i, graphs in enumerate(plots.flat):
        weights = np.ones_like(hists[i])/float(len(hists[i]))
        im = graphs.hist(hists[i], bins=256, range=[-0.5, 1], weights=weights)
        graphs.set_title(titles[i])
    fig.suptitle("Histograms for intensity distribution", fontsize=14)
    plt.savefig(filename+"_histogram.png")
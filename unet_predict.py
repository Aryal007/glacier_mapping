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

import yaml, pathlib, pickle, os
from addict import Dict
import rasterio
import numpy as np
import warnings
import pdb
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib.colors import ListedColormap
from skimage.morphology import remove_small_objects, remove_small_holes
cmap = ListedColormap(["silver", "lightgreen", "forestgreen", "cyan", "tan", "dodgerblue", "white"])

warnings.filterwarnings("ignore")

font = {'size'   : 20}

matplotlib.rc('font', **font)

if __name__ == "__main__":
    classes = ["Wetlands", "Agriculture", "Impervious", "Water", "Soil", "Forest", "Snow"]
    labels = [4, 2, 1, 6, 5, 3, 7]
    _threshold = [3, 3, 3, 3, 3, 3, 3]
    #classes = ["Snow"]
    #labels = [7]
    #_threshold = [15]

    vals = ["15N/1973_3709_13", "35N/4666_2369_13", "47N/6353_3661_13",
            "21S/2789_4694_13", "50N/6761_3129_13", "30N/4062_3943_13",
            "49N/6678_3579_13", "43N/5830_3834_13", "56S/7517_4908_13",
            "19S/2569_4513_13"]
    conf = Dict(yaml.safe_load(open('./conf/unet_predict.yaml')))
    data_dir = pathlib.Path(conf.data_dir)

    models = []
    for c in classes:
        model_path = data_dir / "runs_dropout" / c / "models" / "model_final.pt"
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        models.append(state_dict)

    loss_fn = fn.get_loss(conf.model_opts.args.outchannels)        
    frame = Framework(
        loss_fn = loss_fn,
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts
    )

    for zone in sorted(os.listdir(data_dir / conf.satellite)):
        if os.path.isdir(data_dir / conf.satellite / zone):
            for cubeid in sorted(os.listdir(data_dir / conf.satellite / zone)):
                if os.path.isdir(data_dir / conf.satellite / zone / cubeid):
                    for tiff in sorted(os.listdir(data_dir / conf.satellite / zone / cubeid / 'L3H-SR' )):
                        #if zone+'/'+cubeid in vals:
                        if tiff.endswith('01.tif'):
                            tif_path = data_dir / conf.satellite / zone / cubeid / 'L3H-SR' / tiff
                            _tiff = rasterio.open(tif_path)
                            tiff_np = _tiff.read()
                            tiff_np = tiff_np.transpose(1, 2, 0)
                            tiff_np = (tiff_np - np.min(tiff_np, axis=(0,1)))/(np.max(tiff_np, axis=(0,1)) - np.min(tiff_np, axis=(0,1)))
                            if conf["add_ndvi"]:
                                tiff_np = add_index(tiff_np, index1 = 3, index2 = 0, comment = "ndvi")
                            if conf["add_ndwi"]:
                                tiff_np = add_index(tiff_np, index1 = 1, index2 = 3, comment = "ndwi")
                            if conf["add_ndswi"]:
                                tiff_np = add_index(tiff_np, index1 = 3, index2 = 2, comment = "ndswi")
                            if conf["add_evi2"]:
                                evi2 = 2.5 * (tiff_np[:,:,3] - tiff_np[:,:,2]) / (tiff_np[:,:,3] + (2.4 * tiff_np[:,:,2]) + 1)
                                tiff_np = np.concatenate((tiff_np, np.expand_dims(evi2, axis=2)), axis=2)
                            if conf["add_osavi1"]:
                                osavi1 = (tiff_np[:,:,3] - tiff_np[:,:,2]) / (tiff_np[:,:,3] + tiff_np[:,:,2] + 0.16)
                                tiff_np = np.concatenate((tiff_np, np.expand_dims(osavi1, axis=2)), axis=2)
                            tiff_np = tiff_np[:,:,conf.use_channels]
                            filename = zone+"_"+cubeid+"_"+tiff.split(".")[0]
                            arr = np.load(data_dir / "processed" / "normalize.npy")
                            mean, std = arr[0][conf.use_channels], arr[1][conf.use_channels]
                            tiff_np = (tiff_np - mean) / std
                            x = np.expand_dims(tiff_np, axis=0)
                            x = torch.from_numpy(x).float() 
                            for i, model in enumerate(models):
                                frame.load_state_dict(model)
                                prediction = frame.infer(x)
                                prediction = torch.nn.Softmax(3)(prediction)
                                prediction = np.asarray(prediction.cpu()).squeeze()
                                prediction = prediction[:,:,0]
                                if i == 0:
                                    _temp_output = prediction > 0.999
                                    _temp_output = remove_small_objects(_temp_output, min_size = _threshold[i], connectivity=3)
                                    _temp_output = remove_small_holes(_temp_output, area_threshold = 50, connectivity=1)
                                    output = (_temp_output).astype(np.uint8)*labels[i]
                                else:
                                    _temp_output = prediction > 0.999
                                    _temp_output = remove_small_objects(_temp_output, min_size = _threshold[i], connectivity=3)
                                    _temp_output = remove_small_holes(_temp_output, area_threshold = 50, connectivity=1)
                                    output[_temp_output] = labels[i]
                            print(f"Finished predicting {filename}")
                            np.save("./changes/"+filename, output)
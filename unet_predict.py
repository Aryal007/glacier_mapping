#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 2:57:34 2021

@author: mibook
"""
from coastal_mapping.data.data import fetch_loaders
from coastal_mapping.data.slice import add_index
from coastal_mapping.model.frame import Framework
import coastal_mapping.model.functions as fn
from coastal_mapping.model.metrics import *

import yaml, pathlib, pickle, os
from tifffile import imread
from addict import Dict
import rasterio
import numpy as np
import warnings, glob
import pdb
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from skimage.morphology import remove_small_objects, remove_small_holes

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    conf = Dict(yaml.safe_load(open('./conf/unet_predict.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    model_path = data_dir / conf.run_name / "floodwater_11_7_8" / "models" / "best" / "model_best.h5"
    tps, fps, fns = 0, 0, 0

    output_dir = conf.run_name.replace("runs", "outputs")
    output_dir = data_dir / output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    input_dir = data_dir / "processed" / "val" / "tiff_*.npy"
    labels_dir = data_dir / "train_labels" 
    input_files = glob.glob(str(input_dir))

    if torch.cuda.is_available():
        model = torch.load(model_path)
    else:
        model = torch.load(model_path, map_location="cpu")
    
    loss_fn = fn.get_loss(conf.model_opts.args.outchannels)        
    frame = Framework(
        loss_fn = loss_fn,
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts
    )

    for inp in input_files:
        fname = inp.split("/")[-1].split(".")[0].split("_")[-1]
        _fname = fname + ".tif"
        gt = imread(labels_dir / _fname)
        inp = np.load(inp)
        mean = np.asarray([0.5259015, 0.58692276, 1.1128232, 0.9134505, 236.96355, 3.4183776, 8.270104, 12.422075, 3.9184833, 0.58966357, 151.06439])
        std = np.asarray([0.09099284, 0.09632826, 0.16587685, 0.69075435, 20.920994, 0.19825858, 6.3861856, 12.3026, 0.94436866, 0.9797459, 17.214708])
        inp = (inp - mean) / std
        x = np.expand_dims(inp, axis=0)
        x = torch.from_numpy(x).float() 
        frame.load_state_dict(model)
        prediction = frame.infer(x)
        prediction = torch.nn.Softmax(3)(prediction)
        prediction = np.asarray(prediction.cpu()).squeeze()
        prediction = prediction[:,:,1]
        output_fname = output_dir / fname
        np.save(output_fname, prediction)
        plt.clf()
        figure, axis = plt.subplots(1, 3)
        axis[0].imshow(prediction)
        axis[0].set_title("Prediction prob")
        axis[1].imshow(gt)
        axis[1].set_title("Ground truth")
        pred_binary = (prediction > 0.4).astype(np.uint8)
        axis[2].imshow(pred_binary)
        axis[2].set_title("Threshold")
        plt.show()
        fig_fname = str(output_fname)+".png" 
        plt.savefig(fig_fname)
        tp, fp, fn = tp_fp_fn(pred_binary, gt, label=1)
        tps += tp
        fps += fp
        fns += fn
        #print(f"Finished predicting {fname}")
    print(f"IoU = {IoU(tps, fps, fns)}")
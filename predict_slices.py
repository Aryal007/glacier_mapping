#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Mar 8 15:15:04 2021

@author: mibook
"""
import numpy as np
import torch
from coastal_mapping.data.slice import remove_and_create
from coastal_mapping.model.frame import Framework
import coastal_mapping.model.functions as fn
import yaml, json, pathlib, os, pickle
from addict import Dict
import matplotlib.pyplot as plt

def predict_proba(x, model):
    y_hat = model.predict_proba(x)
    return y_hat

if __name__ == "__main__":
    conf = Dict(yaml.safe_load(open('./conf/predict_slices.yaml')))
    processed_dir = pathlib.Path(conf.processed_dir)
    out_dir = pathlib.Path(conf.out_dir)
    print(f"Loading {conf.model} model")
    if conf.model == "ndwi" or conf.model == "ndswi":
        remove_and_create(out_dir / conf.model)
        images_np = [x for x in os.listdir(processed_dir) if "tiff" in x]
        print(f"Number of files: {len(images_np)}")
        for image_np in images_np:
            savename = (image_np.split(".")[0]).replace("tiff", "mask")
            if conf.model == "ndwi":
                np.save(out_dir / conf.model / savename, np.load(processed_dir / image_np)[:,:,5])
            elif conf.model == "ndswi":
                np.save(out_dir / conf.model / savename, np.load(processed_dir / image_np)[:,:,6])
            else:
                raise ValueError("Wrong model")
        exit(0)
    print("Not remote sensing indices")
    try:
        # Load random_forest or xgboost model
        model_dir = pathlib.Path(conf.model_dir) / "ml_data" / conf.model
        model = pickle.load(open( model_dir / "estimator.sav", 'rb'))
        remove_and_create(out_dir / conf.model)
    except:
        # Load U-Net model
        model_path = pathlib.Path(conf.model_dir) / "runs" / conf.run_name / "models" / "model_final.pt"
        loss_fn = fn.get_loss(conf.model_opts.args.outchannels)
        frame = Framework(
            loss_fn = loss_fn,
            model_opts=conf.model_opts
        )
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        frame.load_state_dict(state_dict)
        remove_and_create(out_dir / conf.run_name)
        arr = np.load(pathlib.Path(conf.model_dir) / "processed" / "normalize.npy")
        mean, std = arr[0][conf.use_channels], arr[1][conf.use_channels]
    
    images_np = [x for x in os.listdir(processed_dir) if "tiff" in x]
    print(f"Number of files: {len(images_np)}")

    for image_np in images_np:
        try:
            savename = (image_np.split(".")[0]).replace("tiff", "mask")
            x = np.load(processed_dir / image_np)
            x = x[:,:,conf.use_channels]
            if "unet" not in conf.model:
                _x = x.reshape((x.shape[0]*x.shape[1]), x.shape[2])
                y_hat = predict_proba(_x, model)
                y_hat = y_hat.reshape(x.shape[0], x.shape[1], y_hat.shape[1])
                y_hat = y_hat[:,:,1]
                np.save(out_dir / conf.model / savename, y_hat)
            else:
                x = (x - mean) / std
                x = np.expand_dims(x, axis=0)
                x = torch.from_numpy(x).float()
                y_hat = frame.infer(x)
                y_hat = torch.nn.Softmax(3)(y_hat)
                y_hat = np.asarray(y_hat.cpu()).squeeze()[:,:,1]
                np.save(out_dir / conf.run_name / savename, y_hat)
        except Exception as e:
            print(e)
            print(image_np)
            pass

    print("Prediction slices completed")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 2:57:34 2021

@author: Aryal007
"""
from segmentation.model.frame import Framework
from segmentation.data.slice import add_index
import segmentation.model.functions as fn

import yaml
import pathlib
import rasterio
import torch
import pdb
from addict import Dict
import numpy as np
import pandas as pd
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    labels_dict = {"Clean Ice": 1, "Debris": 2}
    conf = Dict(yaml.safe_load(open('./conf/predict_slices.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    image_dir = data_dir / "Landsat7_2005"
    label_dir = data_dir / "labels"
    df = pd.read_csv(image_dir / 'metadata.csv')
    test_df = df[df.split == "test"]
    image_ids = sorted(list(set(test_df.image_id)))
    image_ids = [i + ".tif" for i in image_ids]
    tiff_path = data_dir / "Landsat7_2005"
    model_path = data_dir / conf.processed_dir / conf.folder_name / \
        conf.run_name / "models" / "model_best.pt"

    loss_fn = fn.get_loss(conf.model_opts.args.outchannels)
    frame = Framework(
        loss_fn=loss_fn,
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        device=(int(conf.gpu_rank))
    )
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")
    frame.load_state_dict(state_dict)

    for image_id in image_ids[6:]:
        print(f"Working with file {image_id}")
        tiff = rasterio.open(tiff_path / image_id)
        tiff_np = tiff.read().transpose(1, 2, 0).astype(np.float32)
        #tiff_np = tiff_np[1500:2500, 3500:4500, :]
        mask = np.sum(tiff_np[:, :, :3], axis=2) == 0
        tiff_np = add_index(tiff_np, index1=3, index2=2)
        tiff_np = add_index(tiff_np, index1=3, index2=4)
        tiff_np = add_index(tiff_np, index1=1, index2=4)
        rgb_img = tiff_np[:, :, :3] / 255
        tiff_np = tiff_np[:, :, conf.use_channels]

        normalize = np.load(data_dir / conf.processed_dir / "normalize_train.npy")
        _mean, _std = normalize[0][conf.use_channels], normalize[1][conf.use_channels]
        
        tiff_np = (tiff_np - _mean) / _std
        x = np.expand_dims(tiff_np, axis=0)
        y = np.zeros((tiff_np.shape[0], tiff_np.shape[1]))
        x = torch.from_numpy(x).float()

        for row in range(0, x.shape[1], conf.window_size[0]):
            for column in range(0, x.shape[2], conf.window_size[1]):
                current_slice = x[:, row:row + conf["window_size"][0], column:column + conf["window_size"][1], :]
                if current_slice.shape[1] != conf.window_size[0] or current_slice.shape[2] != conf.window_size[1]:
                    temp = np.zeros((1, conf.window_size[0], conf.window_size[1], x.shape[3]))
                    temp[:,:current_slice.shape[1],:current_slice.shape[2],:] = current_slice
                    current_slice = torch.from_numpy(temp).float()
                    _mask = mask[:current_slice.shape[1],:current_slice.shape[2]]
                prediction = frame.infer(current_slice)
                prediction = torch.nn.Softmax(3)(prediction)
                prediction = np.squeeze(prediction.cpu())
                _prediction = np.zeros(
                    (prediction.shape[0], prediction.shape[1]))
                for k, v in labels_dict.items():
                    _prediction[prediction[:, :, v] >= conf.threshold[v-1]] = v
                _prediction = _prediction+1
                prediction = _prediction
                endrow_dest = row + conf.window_size[0]
                endrow_source = conf.window_size[0]
                endcolumn_dest = column + conf.window_size[0]
                endcolumn_source = conf.window_size[1]
                if endrow_dest > y.shape[0]:
                    endrow_source = y.shape[0] - row
                    endrow_dest = y.shape[0]
                if endcolumn_dest > y.shape[1]:
                    endcolumn_source = y.shape[1] - column
                    endcolumn_dest = y.shape[1]
                try:
                    y[row:endrow_dest,
                      column:endcolumn_dest] = prediction[0:endrow_source,
                                                          0:endcolumn_source]
                except Exception as e:
                    print(e)
                    print("Something wrong with indexing!")
        y[mask] = 0
        plt.imshow(rgb_img)
        plt.savefig("image.png")
        plt.imshow((y==1).astype(np.uint8))
        plt.colorbar()
        plt.savefig("background.png")
        plt.imshow((y==2).astype(np.uint8))
        plt.savefig("cleanice.png")
        plt.imshow((y==3).astype(np.uint8))
        plt.savefig("debris.png")
        pdb.set_trace()

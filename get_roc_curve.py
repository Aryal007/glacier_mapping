#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 2:39:37 2021

@author: Aryal007
"""
from segmentation.data.data import fetch_loaders
from segmentation.model.frame import Framework
import segmentation.model.functions as fn

import yaml
import pathlib
import warnings
import torch
import matplotlib
import os
import matplotlib.pyplot as plt
from addict import Dict
import numpy as np
import pdb
from sklearn.metrics import roc_curve, auc, roc_auc_score

warnings.filterwarnings("ignore")


def min_max_normalize(conf, X):
    data_dir = pathlib.Path(conf.data_dir)
    _min = np.load(data_dir / "normalize_train.npy")[2][conf.use_channels]
    _max = np.load(data_dir / "normalize_train.npy")[3][conf.use_channels]
    X = np.clip(X, _min, _max)
    X = (X - _min) / (_max - _min)
    return X


def mean_std_normalize(conf, X):
    data_dir = pathlib.Path(conf.data_dir)
    _mean = np.load(data_dir / "normalize_train.npy")[0][conf.use_channels]
    _std = np.load(data_dir / "normalize_train.npy")[1][conf.use_channels]
    X = (X - _mean) / _std
    return X


def plot_iou_curve(y, scores, glacier_type):
    def get_iou(y_tp, y_fp, y_fn):
        return y_tp / (y_tp + y_fp + y_fn)

    bins = np.linspace(0, 1, 101)
    ious = np.zeros_like(bins)
    label = 1
    y = y.astype(np.bool)
    for i, threshold in enumerate(bins):
        pred = (scores >= threshold).astype(np.uint8)
        tp = ((pred == label) & (y == label)).sum().item()
        fp = ((pred == label) & (y != label)).sum().item()
        fn = ((pred != label) & (y == label)).sum().item()
        ious[i] = get_iou(tp, fp, fn)
    plt.figure()
    plt.plot(
        bins,
        ious,
        label=f"IOU (best = {np.max(ious):2.2f}, threshold = {bins[np.argmax(ious)]:2.2f})")
    plt.grid()
    plt.xlim([0.0, 1.0])
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title(f"Threshold sweep for best IoU for {glacier_type}")
    plt.legend(loc="lower right")
    plt.savefig(f"iou_{glacier_type}.png")


def plot_roc_curve(y, scores, glacier_type):
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver operating characteristic example {glacier_type}")
    plt.legend(loc="lower right")
    plt.savefig(f"roc_{glacier_type}.png")


if __name__ == "__main__":
    conf = Dict(yaml.safe_load(open('./conf/get_roc_curve.yaml')))
    val_dir = pathlib.Path(conf.data_dir) / "val"
    class_name = conf.class_name

    loss_fn = fn.get_loss(conf.model_opts.args.outchannels)
    frame = Framework(
        loss_fn=loss_fn,
        model_opts=conf.model_opts,
        loss_opts=conf.loss_opts,
        device=conf.gpu_rank
    )

    model_path = f"{conf.data_dir}/runs/{conf.run_name}/models/model_best.pt"
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")
    frame.load_state_dict(state_dict)

    val_files = [f for f in os.listdir(val_dir) if "tiff" in f]
    _arr = np.load(val_dir / val_files[0])

    for glacier_type, glacier_index in conf.class_name.items():
        mask_arr = np.zeros((len(val_files), _arr.shape[0], _arr.shape[1]))
        pred_arr, gt_arr = np.zeros_like(mask_arr), np.zeros_like(mask_arr)
        for i, f in enumerate(val_files):
            X = np.load(val_dir / f)[:, :, conf.use_channels]
            mask_arr[i] = np.sum(X[:, :, :5], axis=2) != 0
            if conf.normalize == "min-max":
                X = min_max_normalize(conf, X)
            elif conf.normalize == "mean-std":
                X = mean_std_normalize(conf, X)
            else:
                raise ValueError("Normalize must be min-max or mean-std")
            X = torch.Tensor(np.expand_dims(X, 0))
            y_pred = frame.act(frame.infer(X))
            y_gt = (
                np.load(
                    val_dir /
                    f.replace(
                        "tiff",
                        "mask")) == glacier_index).astype(
                np.int8)
            pred_arr[i] = np.squeeze(y_pred.cpu())[:, :, glacier_index]
            gt_arr[i] = y_gt
        mask_arr = mask_arr.astype(np.bool)
        y = gt_arr[mask_arr]
        scores = pred_arr[mask_arr]

        if conf.iou_threshold:
            plot_iou_curve(y, scores, glacier_type)
        else:
            plot_roc_curve(y, scores, glacier_type)

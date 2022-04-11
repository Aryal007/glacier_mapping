from segmentation.model.frame import Framework
import segmentation.model.functions as fn

import yaml
import pdb
import os
import pathlib
import torch
from addict import Dict
import numpy as np
import pandas as pd
from segmentation.model.metrics import *
import segmentation.data.slice as sl

def get_tp_fp_fn(pred, true):
    pred, true = torch.from_numpy(pred), torch.from_numpy(true)
    tp, fp, fn = tp_fp_fn(pred, true)
    return tp, fp, fn

def get_precision_recall_iou(tp, fp, fn):
    p, r, i = precision(tp, fp, fn), recall(tp, fp, fn), IoU(tp, fp, fn)
    return p, r, i

if __name__ == "__main__":
    labels_dict = {"Clean Ice": 1, "Debris": 2}
    conf = Dict(yaml.safe_load(open('./conf/unet_predict.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    preds_dir = data_dir / conf.processed_dir / "preds" / conf.run_name
    columns = ["tile_name", "ci_precision", "ci_recall", "ci_IoU", "debris_precision", "debris_recall", "debris_IoU"]
    df = pd.DataFrame(columns=columns)
    sl.remove_and_create(preds_dir)
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

    arr = np.load(data_dir / conf.processed_dir / "normalize_train.npy")
    if conf.normalize == "mean-std":
        _mean, _std = arr[0][conf.use_channels], arr[1][conf.use_channels]
    if conf.normalize == "min-max":
        _min, _max = arr[2][conf.use_channels], arr[3][conf.use_channels]

    files = os.listdir(data_dir / conf.processed_dir / "test")
    inputs = [x for x in files if "tiff" in x]

    ci_tp_sum, ci_fp_sum, ci_fn_sum = 0, 0, 0
    debris_tp_sum, debris_fp_sum, debris_fn_sum = 0, 0, 0

    for x_fname in inputs:
        x = np.load(data_dir / conf.processed_dir / "test" / x_fname)[:,:,conf.use_channels]
        mask = np.sum(x[:,:,:5], axis=2) == 0
        if conf.normalize == "mean-std":
            x = (x - _mean) / _std
        if conf.normalize == "min-max":
            x = (x - _min) / (_max - _min)
        
        y_fname = x_fname.replace("tiff", "mask")
        save_fname = x_fname.replace("tiff", "pred")
        y_true = np.load(data_dir / conf.processed_dir / "test" / y_fname) + 1
        y_true[mask] = 0

        x = torch.from_numpy(np.expand_dims(x, axis=0)).float()
        pred = frame.infer(x)
        pred = torch.nn.Softmax(3)(pred)
        pred = np.squeeze(pred.cpu())
        prob_pred = pred
        prob_pred[mask] = 0
        
        _pred = np.zeros((pred.shape[0], pred.shape[1]))
        for k, v in labels_dict.items():
            _pred[pred[:, :, v] >= conf.threshold[v-1]] = v
        _pred = _pred+1
        _pred[mask] = 0
        y_pred = _pred
        ci_pred, debris_pred = y_pred == 2, y_pred == 3
        ci_true, debris_true = y_true == 2, y_true == 3
        ci_tp, ci_fp, ci_fn = get_tp_fp_fn(ci_pred, ci_true)
        debris_tp, debris_fp, debris_fn = get_tp_fp_fn(debris_pred, debris_true)
        ci_precision, ci_recall, ci_iou = get_precision_recall_iou(ci_tp, ci_fp, ci_fn)
        debris_precision, debris_recall, debris_iou = get_precision_recall_iou(debris_tp, debris_fp, debris_fn)
        _row = [save_fname, ci_precision, ci_recall, ci_iou, debris_precision, debris_recall, debris_iou]
        df = df.append(pd.DataFrame([_row], columns=columns), ignore_index=True)
        np.save(preds_dir / save_fname, prob_pred.numpy())
        ci_tp_sum += ci_tp
        ci_fp_sum += ci_fp
        ci_fn_sum += ci_fn
        debris_tp_sum += debris_tp
        debris_fp_sum += debris_fp
        debris_fn_sum += debris_fn

    ci_precision, ci_recall, ci_iou = get_precision_recall_iou(ci_tp_sum, ci_fp_sum, ci_fn_sum)
    debris_precision, debris_recall, debris_iou = get_precision_recall_iou(debris_tp_sum, debris_fp_sum, debris_fn_sum)
    _row = ["Total", ci_precision, ci_recall, ci_iou, debris_precision, debris_recall, debris_iou]
    df = df.append(pd.DataFrame([_row], columns=columns), ignore_index=True)
    print(f"{dict(zip(columns, _row))}")
    df.to_csv(preds_dir / "metadata.csv")
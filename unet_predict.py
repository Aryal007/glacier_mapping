from segmentation.model.frame import Framework
import segmentation.model.functions as fn

import yaml, pdb, os, pathlib, torch
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
    conf = Dict(yaml.safe_load(open('./conf/unet_predict.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    preds_dir = data_dir / conf.out_processed_dir / "preds" / conf.run_name
    columns = ["tile_name", "ci_precision", "ci_recall", "ci_IoU", "debris_precision", "debris_recall", "debris_IoU"]
    df = pd.DataFrame(columns=columns)

    sl.remove_and_create(preds_dir)

    cleanice_model_path = data_dir / conf.cleanice_processed_dir / conf.folder_name / \
        conf.run_name / "models" / "model_best.pt"
    debris_model_path = data_dir / conf.debris_processed_dir / conf.folder_name / \
        conf.run_name / "models" / "model_best.pt"

    loss_fn = fn.get_loss(conf.model_opts_cleanice.args.outchannels)
    cleanice_frame = Framework(
        loss_fn=loss_fn,
        model_opts=conf.model_opts_cleanice,
        optimizer_opts=conf.optim_opts,
        device=(int(conf.gpu_rank))
    )
    debris_frame = Framework(
        loss_fn=loss_fn,
        model_opts=conf.model_opts_debris,
        optimizer_opts=conf.optim_opts,
        device=(int(conf.gpu_rank))
    )
    if torch.cuda.is_available():
        cleanice_state_dict = torch.load(cleanice_model_path)
        debris_state_dict = torch.load(debris_model_path)
    else:
        cleanice_state_dict = torch.load(cleanice_model_path, map_location="cpu")
        debris_state_dict = torch.load(debris_model_path, map_location="cpu")
    cleanice_frame.load_state_dict(cleanice_state_dict)
    debris_frame.load_state_dict(debris_state_dict)

    arr = np.load(data_dir / conf.cleanice_processed_dir / "normalize_train.npy")
    if conf.normalize == "mean-std":
        _mean, _std = arr[0], arr[1]
    if conf.normalize == "min-max":
        _min, _max = arr[2], arr[3]

    files = os.listdir(data_dir / conf.cleanice_processed_dir / "test")
    inputs = [x for x in files if "tiff" in x]

    ci_tp_sum, ci_fp_sum, ci_fn_sum = 0, 0, 0
    debris_tp_sum, debris_fp_sum, debris_fn_sum = 0, 0, 0

    for x_fname in inputs:
        x = np.load(data_dir / conf.cleanice_processed_dir / "test" / x_fname)
        mask = np.sum(x, axis=2) == 0
        if conf.normalize == "mean-std":
            x = (x - _mean) / _std
        if conf.normalize == "min-max":
            x = (x - _min) / (_max - _min)
        
        y_fname = x_fname.replace("tiff", "mask")
        save_fname = x_fname.replace("tiff", "pred")
        y_cleanice_true = np.load(data_dir / conf.cleanice_processed_dir / "test" / y_fname) + 1
        y_debris_true = np.load(data_dir / conf.debris_processed_dir / "test" / y_fname) + 1
        y_cleanice_true, y_debris_true = y_cleanice_true[~mask], y_debris_true[~mask]

        _x = torch.from_numpy(np.expand_dims(x[:,:,conf.use_channels_cleanice], axis=0)).float()
        pred_cleanice = cleanice_frame.infer(_x)
        pred_cleanice = torch.nn.Softmax(3)(pred_cleanice)
        pred_cleanice = np.squeeze(pred_cleanice.cpu())
        _x = torch.from_numpy(np.expand_dims(x[:,:,conf.use_channels_debris], axis=0)).float()
        pred_debris = debris_frame.infer(_x)
        pred_debris = torch.nn.Softmax(3)(pred_debris)
        pred_debris = np.squeeze(pred_debris.cpu())
        _pred = np.zeros((pred_debris.shape[0], pred_debris.shape[1]))
        _pred[pred_cleanice[:, :, 1] >= conf.threshold[0]] = 1
        _pred[pred_debris[:, :, 1] >= conf.threshold[1]] = 2
        _pred = _pred+1
        _pred[mask] = 0
        y_pred = _pred
        y_pred_prob = np.zeros((pred_debris.shape[0], pred_debris.shape[1], 3))
        y_pred_prob[:,:,1] = pred_cleanice[:, :, 1]
        y_pred_prob[:,:,2] = pred_debris[:, :, 1]
        y_pred_prob[:,:,0] = np.min(np.concatenate((pred_cleanice[:, :, 0][:,:, None], pred_debris[:, :, 0][:,:,None]), axis=2), axis=2)
        y_pred_prob[mask] = 0
        y_pred = y_pred[~mask]
        ci_pred, debris_pred = (y_pred == 2).astype(np.int8), (y_pred == 3).astype(np.int8)
        ci_true, debris_true = (y_cleanice_true == 2).astype(np.int8), (y_debris_true == 2).astype(np.int8)
        ci_tp, ci_fp, ci_fn = get_tp_fp_fn(ci_pred, ci_true)
        debris_tp, debris_fp, debris_fn = get_tp_fp_fn(debris_pred, debris_true)
        ci_precision, ci_recall, ci_iou = get_precision_recall_iou(ci_tp, ci_fp, ci_fn)
        debris_precision, debris_recall, debris_iou = get_precision_recall_iou(debris_tp, debris_fp, debris_fn)
        _row = [save_fname, ci_precision, ci_recall, ci_iou, debris_precision, debris_recall, debris_iou]
        df = df.append(pd.DataFrame([_row], columns=columns), ignore_index=True)
        np.save(preds_dir / save_fname, y_pred_prob)
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
import numpy as np
import pandas as pd
import os, pdb, glob, torch, sys, yaml
from addict import Dict
from tqdm import tqdm
from loguru import logger
from tifffile import imread
from codeexecution.assets.frame import Framework
from coastal_mapping.model.metrics import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, remove_small_holes

val_ids = ['kno',  'psx', 'qxb', 'tht']
model_id = "I"
out_dir = "./validate"
threshold = 0.65

def get_model(model_id: str):
    """
    Return model parameter and weights
    """
    conf = Dict(yaml.safe_load(open('./codeexecution/assets/conf.yaml'))) 
    model_path = f"./codeexecution/assets/model_{model_id}.h5"
    frame = Framework(model_opts=conf.model_opts)
    state_dict = torch.load(model_path, map_location="cpu")
    frame.load_state_dict(state_dict)
    return frame


def get_inp_image(chip_id: str):
    """
    Given an image ID, return numpy image for prediction
    """
    arr_vh = imread(f"./data/train_features/{chip_id}_vh.tif")
    arr_vv = imread(f"./data/train_features/{chip_id}_vv.tif")
    arr_change = imread(f"./data/train_features/{chip_id}_jrc-gsw-change.tif")
    arr_extent = imread(f"./data/train_features/{chip_id}_jrc-gsw-extent.tif")
    arr_occurrence = imread(f"./data/train_features/{chip_id}_jrc-gsw-occurrence.tif")
    arr_recurrence = imread(f"./data/train_features/{chip_id}_jrc-gsw-recurrence.tif")
    arr_seasonality = imread(f"./data/train_features/{chip_id}_jrc-gsw-seasonality.tif")
    arr_transitions = imread(f"./data/train_features/{chip_id}_jrc-gsw-transitions.tif")
    arr_nasadem = imread(f"./data/train_features/{chip_id}_nasadem.tif")
    # TODO: this is the main work! it is your job to implement this
    vv = np.expand_dims(gaussian(arr_vv, sigma=1.5), axis=2)
    vv_minmax = (vv - np.min(vv)) / (np.max(vv) - np.min(vv))
    vh = np.expand_dims(gaussian(arr_vh, sigma=1.5), axis=2)
    vh_minmax = (vh - np.min(vh)) / (np.max(vh) - np.min(vh))
    add_minmax = vv_minmax + vh_minmax
    mul_minmax = vv_minmax * vh_minmax
    nprb = (vv_minmax - vh_minmax + 2) / (vv_minmax + vh_minmax + 2)
    change = np.expand_dims(arr_change, axis=2)
    extent = np.expand_dims(arr_extent, axis=2)
    occurrence = np.expand_dims(arr_occurrence, axis=2)
    recurrence = np.expand_dims(arr_recurrence, axis=2)
    seasonality = np.expand_dims(arr_seasonality, axis=2)
    transitions = np.expand_dims(arr_transitions, axis=2)
    nasadem = np.expand_dims(arr_nasadem, axis=2)
    inp =  np.concatenate((vv,vh,vv_minmax,vh_minmax,add_minmax,mul_minmax,nprb,change,extent,occurrence,recurrence,seasonality,transitions,nasadem), axis=2)
    mean = np.asarray([-10.775323, -17.547888, 0.50388277, 0.52909607, 1.0329787, 0.29140142, 0.66001433, 236.96355, 3.4183776, 8.270104, 12.422075, 3.9184833, 0.58966357, 151.06439])
    std = np.asarray([2.8078806, 3.145529, 0.11285119, 0.1308618, 0.22676097, 0.11502454, 0.06593883, 20.920994, 0.19825858, 6.3861856, 12.3026, 0.94436866, 0.9797459, 17.214708])
    inp = (inp - mean) / std
    x = np.expand_dims(inp, axis=0)
    x = torch.from_numpy(x).float()
    return x


def get_gt(chip_id: str):
    y = imread(f"./data/train_labels/{chip_id}.tif")
    return torch.from_numpy(y)


def get_chip_ids():
    location = "./data/train_labels/*.tif"
    _files = glob.glob(location)
    ids = []
    for f in _files:
        for val_id in val_ids:
            if val_id in f:
                ids.append(f.split("/")[-1].split(".")[0])
    return ids

def get_prediction(frame, x):
    """
    Given model and numpy image, return numpy image for prediction
    """
    prediction = frame.infer(x)
    prediction = torch.nn.Softmax(3)(prediction)
    prediction = np.asarray(prediction.cpu()).squeeze()
    prediction = prediction[:,:,1]
    return torch.from_numpy(prediction)

if __name__ == "__main__":
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    model = get_model(model_id)
    ids = get_chip_ids()
    tp, fp, fn = 0, 0, 0
    y_test, y_score = np.asarray([]), np.asarray([])
    df = pd.DataFrame(columns = ['chip_id', 'tp', 'fp', 'fn', 'precision', 'recall', 'dice', 'IoU'])
    for chip_id in tqdm(ids, miniters=25, file=sys.stdout, leave=True):
        logger.info(f"Generating predictions for {chip_id} ...")
        x = get_inp_image(chip_id)
        y = get_gt(chip_id)
        pred = get_prediction(model, x)
        pred[y == 255] = 0
        y[y == 255] = 0
        y_test = np.concatenate((y_test, y.flatten().numpy()))
        y_score = np.concatenate((y_score, pred.flatten().numpy()))
        pred = pred > threshold
        pred = pred.numpy()
        pred = remove_small_objects(pred, min_size = 5, connectivity=2)
        pred = remove_small_holes(pred, area_threshold = 5, connectivity=2)
        pred = torch.from_numpy(pred).type(torch.int8)
        _tp, _fp, _fn = tp_fp_fn(pred, y)
        _recall = recall(_tp, _fp, _fn)
        _dice = dice(_tp, _fp, _fn)
        _precision = precision(_tp, _fp, _fn)
        _IoU = IoU(_tp, _fp, _fn)
        new_row = {'chip_id':chip_id, 'tp':_tp, 'fp':_fp, 'fn':_fn,
                   'precision':_precision, 'recall':_recall, 'dice':_dice, 'IoU':_IoU}
        df = df.append(new_row, ignore_index=True)
        tp += _tp
        fp += _fp
        fn += _fn
    new_row = {'chip_id':'Average', 'tp':0, 'fp':0, 'fn':0,
                'precision':np.mean(df.precision), 'recall':np.mean(df.recall), 'dice':np.mean(df.dice), 'IoU':np.mean(df.IoU)}
    df = df.append(new_row, ignore_index=True)
    _recall = recall(tp, fp, fn)
    _dice = dice(tp, fp, fn)
    _precision = precision(tp, fp, fn)
    _IoU = IoU(tp, fp, fn)
    new_row = {'chip_id':'Total', 'tp':tp, 'fp':fp, 'fn':fn,
                'precision':_precision, 'recall':_recall, 'dice':_dice, 'IoU':_IoU}
    df = df.append(new_row, ignore_index=True)
    df.to_csv(f'./validate/model_{model_id}.csv')
    prec, recall, _ = precision_recall_curve(y_test, y_score,
                                         pos_label=1)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    plt.savefig(f'./validate/model_{model_id}.png')
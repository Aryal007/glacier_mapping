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
from skimage.morphology import disk
from skimage.filters import median
from skimage import exposure
from skimage.morphology import remove_small_objects, remove_small_holes

val_ids = ['pxs',  'jja', 'qxb']
#val_ids = ['kuo',  'tht', 'qus', 'coz', 'awc']
#val_ids = ['hbe',  'tnp', 'wvy', 'ayt']
model_id = "A"
out_dir = "./validate"
threshold = 0.5
min_values = np.array([-33.510303, -39.171803, -182.45174])
max_values = np.array([7.2160087, 2.8161404, 40.3697])
supplementary_min_values = np.array([0, 0, 0, 0, 0, 0, 0])
supplementary_max_values = np.array([255, 255, 255, 255, 255, 1, 1000])

def get_image(vv, vh, smooth=False):
    if smooth:
        vv = median(vv, disk(2))
        vh = median(vh, disk(2))
    blue = np.nan_to_num(vv / vh)
    img = np.concatenate((np.clip(vv[:,:,None], min_values[0], max_values[0]), 
                        np.clip(vh[:,:,None], min_values[1], max_values[1]), 
                        np.clip(blue[:,:,None], min_values[2], max_values[2])), 
                        axis=2)
    return img

def add_rsi(img):
    out = np.zeros((img.shape[0], img.shape[1], 6))
    out[:,:,:3] = img
    vv = img[:,:,0]
    vh = img[:,:,1]
    blue = img[:,:,2]
    nprb = np.clip(np.nan_to_num((vv - vh) / (vv + vh)), -1, 1)
    bi = np.sqrt(np.square(vv)+np.square(vh)+np.square(blue))/3
    ndwi = np.clip(np.nan_to_num((blue - vv - vh) / (blue + vv + vh)), -1, 1)
    out[:,:,3] = nprb
    out[:,:,4] = bi
    out[:,:,5] = ndwi
    return out

def get_model(model_id: str):
    """
    Return model parameter and weights
    """
    conf = Dict(yaml.safe_load(open('./codeexecution/assets/conf.yaml'))) 
    model_path = f"./codeexecution/assets/model_{model_id}.pt"
    frame = Framework(model_opts=conf.model_opts)
    state_dict = torch.load(model_path, map_location="cpu")
    frame.load_state_dict(state_dict)
    return frame

def add_supplementary(img, chip_id):
    n_channels = img.shape[2]
    out = np.zeros((img.shape[0], img.shape[1], n_channels+7))
    out[:,:,:n_channels] = img
    change = np.clip(np.nan_to_num(imread(f"./data/train_features/{chip_id}_jrc-gsw-change.tif")), supplementary_min_values[0], supplementary_max_values[0])[:,:,None]
    extent = np.clip(np.nan_to_num(imread(f"./data/train_features/{chip_id}_jrc-gsw-extent.tif")), supplementary_min_values[1], supplementary_max_values[1])[:,:,None]
    occurrence = np.clip(np.nan_to_num(imread(f"./data/train_features/{chip_id}_jrc-gsw-occurrence.tif")), supplementary_min_values[2], supplementary_max_values[2])[:,:,None]
    recurrence = np.clip(np.nan_to_num(imread(f"./data/train_features/{chip_id}_jrc-gsw-recurrence.tif")), supplementary_min_values[3], supplementary_max_values[3])[:,:,None]
    seasonality = np.clip(np.nan_to_num(imread(f"./data/train_features/{chip_id}_jrc-gsw-seasonality.tif")), supplementary_min_values[4], supplementary_max_values[4])[:,:,None]
    transitions = np.clip(np.nan_to_num(imread(f"./data/train_features/{chip_id}_jrc-gsw-transitions.tif")), supplementary_min_values[5], supplementary_max_values[5])[:,:,None]
    nasadem = np.clip(np.nan_to_num(imread(f"./data/train_features/{chip_id}_nasadem.tif")), supplementary_min_values[6], supplementary_max_values[6])[:,:,None]
    supplementary = np.concatenate((change, extent, occurrence, recurrence, seasonality, transitions, nasadem), axis=2)
    supplementary = (supplementary - supplementary_min_values) / (supplementary_max_values - supplementary_min_values)
    out[:,:,n_channels:] = supplementary
    return out

def remove_outliers(img):
    mean = np.mean(img, axis=(0,1))
    std = np.std(img, axis=(0,1))
    img = np.clip(img, mean-3*std, mean+3*std)
    return img

def min_max(img):
    img = (img - min_values) / (max_values - min_values)
    img[:,:,2] = img[:,:,2]/1.5
    return img

def get_inp_image(chip_id: str):
    """
    Given an image ID, return numpy image for prediction
    """
    vv = imread(f"./data/train_features/{chip_id}_vv.tif")
    vh = imread(f"./data/train_features/{chip_id}_vh.tif")
    img = get_image(vv, vh, smooth=True)
    img = remove_outliers(img)
    img = min_max(img)
    img = add_rsi(img)
    img = add_supplementary(img, chip_id)
    x = np.expand_dims(img, axis=0)
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
        pred = get_prediction(model, x)
        y = get_gt(chip_id)
        pred[y == 255] = 0
        y[y == 255] = 0
        orig_pred = pred
        y_test = np.concatenate((y_test, y.flatten().numpy()))
        y_score = np.concatenate((y_score, pred.flatten().numpy()))
        pred = pred.numpy()
        pred = pred > threshold
        pred = remove_small_holes(pred, area_threshold = 50, connectivity=2)
        pred = remove_small_objects(pred, min_size = 50, connectivity=2)    
        fig, ax = plt.subplots(2,2)
        ax[0,0].set_title('RGB Image')
        ax[0,0].imshow(x.numpy()[0][:,:,:3])
        ax[0,1].set_title('Labels')
        ax[0,1].imshow(y, cmap="gray")
        ax[1,0].set_title('Prediction')
        ax[1,0].imshow(orig_pred, cmap="gray")
        ax[1,1].set_title('Processed prediction')
        ax[1,1].imshow(pred, cmap="gray")
        ax[0,0].axis('off')
        ax[0,1].axis('off')
        ax[1,0].axis('off')
        ax[1,1].axis('off')
        pred = torch.from_numpy(pred).type(torch.int8)
        #pred = pred.type(torch.int8)
        _tp, _fp, _fn = tp_fp_fn(pred, y)
        _recall = recall(_tp, _fp, _fn)
        _dice = dice(_tp, _fp, _fn)
        _precision = precision(_tp, _fp, _fn)
        _IoU = IoU(_tp, _fp, _fn)
        fig.suptitle(f'Dice: {_dice:.3f}, IoU: {_IoU:.3f}')
        plt.savefig(f"./validate/images/{chip_id}_{model_id}.png")
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
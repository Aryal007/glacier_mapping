#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 23:18:43 2020

@author: mibook

Training/Validation Functions
"""
import numpy as np
from tqdm import tqdm
from .metrics import *
import logging, datetime, pdb, torch
from torchvision.utils import make_grid
from segmentation.model.losses import *

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
HANDLER = logging.StreamHandler()
HANDLER.setLevel(logging.INFO)
FORMATTER = logging.Formatter("%(message)s")
LOGGER.addHandler(HANDLER)

def log(level, message):
    """Log the message at a given level (from the standard logging package levels: ERROR, INFO, DEBUG etc).
    Add a datetime prefix to the log message, and a SystemLog: prefix provided it is public data.
    
    Args:
        level (int): logging level, best set by using logging.(INFO|DEBUG|WARNING) etc
        message (str): mesage to log
    """
    message = "{}\t{}\t{}".format(datetime.datetime.now().strftime('%d-%m-%Y, %H:%M:%S'), logging._levelToName[level], message)
    message = "SystemLog: " + message
    logging.log(level, message)

def train_epoch(epoch, loader, frame, conf):
    """Train model for one epoch

    This makes one pass through a dataloader and updates the model in the
    associated frame.

    :param loader: A pytorch DataLoader containing x,y pairs
      with which to train the model.
    :type loader: torch.data.utils.DataLoader
    :param frame: A Framework object wrapping both the model and the
      optimization setup.
    :type frame: Framework
    :param metrics_opts: A dictionary whose keys specify which metrics to
      compute on the predictions from the model.
    :type metrics_opts: dict
    :return (train_loss, metrics): A tuple containing the average epoch loss
      and the metrics on the training set.
    """
    metrics_opts, masked, n_classes, grad_accumulation_steps = conf.metrics_opts, conf.loss_masked, len(conf.log_opts.mask_names), conf.grad_accumulation_steps
    loss, batch_loss, tp, fp, fn = 0, 0, torch.zeros(n_classes), torch.zeros(n_classes), torch.zeros(n_classes)
    train_iterator = tqdm(loader, desc="Train Iter (Epoch=X Steps=X loss=X.XXX lr=X.XXXXXXX)")
    for i, (x,y) in enumerate(train_iterator):
        y_hat, _loss = frame.optimize(x, y)
        if grad_accumulation_steps != "None":
            if (i+1) % grad_accumulation_steps == 0:
                frame.zero_grad()
        else:
            frame.zero_grad()
        loss += _loss.item()
        y_hat = frame.segment(y_hat)
        _tp, _fp, _fn = frame.metrics(y_hat, y, masked)
        tp += _tp
        fp += _fp 
        fn += _fn 
        train_iterator.set_description("Train, Epoch=%d Steps=%d Loss=%5.3f Avg_Loss=%5.3f " %(epoch, i, _loss.item(), loss/(i+1)))
    frame.zero_grad()
    metrics = get_metrics(tp, fp, fn, metrics_opts)
    return loss / len(loader.dataset), metrics


def validate(epoch, loader, frame, conf):
    """Compute Metrics on a Validation Loader

    To honestly evaluate a model, we should compute its metrics on a validation
    dataset. This runs the model in frame over the data in loader, compute all
    the metrics specified in metrics_opts.

    :param loader: A DataLoader containing x,y pairs with which to validate the
      model.
    :type loader: torch.utils.data.DataLoader
    :param frame: A Framework object wrapping both the model and the
      optimization setup.
    :type frame: Framework
    :param metrics_opts: A dictionary whose keys specify which metrics to
      compute on the predictions from the model.
    :type metrics_opts: dict
    :return (val_loss, metrics): A tuple containing the average validation loss
      and the metrics on the validation set.
    """
    metrics_opts, masked, n_classes = conf.metrics_opts, conf.loss_masked, len(conf.log_opts.mask_names)
    loss, tp, fp, fn = 0, torch.zeros(n_classes), torch.zeros(n_classes), torch.zeros(n_classes)
    val_iterator = tqdm(loader, desc="Val Iter (Epoch=X Steps=X loss=X.XXX lr=X.XXXXXXX)")
    channel_first = lambda x: x.permute(0, 3, 1, 2)
    for i, (x,y) in enumerate(val_iterator):
        with torch.no_grad():
            y_hat = frame.infer(x)
            _loss = frame.calc_loss(channel_first(y_hat), channel_first(y)).item()
            loss += _loss
            y_hat = frame.segment(y_hat)
            _tp, _fp, _fn = frame.metrics(y_hat, y, masked)
            tp += _tp 
            fp += _fp 
            fn += _fn 
            val_iterator.set_description("Val,   Epoch=%d Steps=%d Loss=%5.3f Avg_Loss=%5.3f " %(epoch, i, _loss, loss/(i+1)))
    frame.val_operations(loss/len(loader.dataset))
    metrics = get_metrics(tp, fp, fn, metrics_opts)  
    return loss / len(loader.dataset), metrics

def log_metrics(writer, metrics, epoch, stage, mask_names=None):
    """Log metrics for tensorboard
    A function that logs metrics from training and testing to tensorboard
    Args:
        writer(SummaryWriter): The tensorboard summary object
        metrics(Dict): Dictionary of metrics to record
        avg_loss(float): The average loss across all epochs
        epoch(int): Total number of training cycles
        stage(String): Train/Val
        mask_names(List): Names of the mask(prediction) to log mmetrics for
    """
    for k, v in metrics.items():
        for name, metric in zip(mask_names, v):
            writer.add_scalar(f"{name}_{str(k)}/{stage}", metric, epoch)

def log_images(writer, frame, batch, epoch, stage):
    """Log images for tensorboard

    Args:
        writer (SummaryWriter): The tensorboard summary object
        frame (Framework): The model to use for inference
        batch (tensor): The batch of samples on which to make predictions
        epoch (int): Current epoch number
        stage (string): specified pipeline stage

    Return:
        Images Logged onto tensorboard
    """
    batch = next(iter(batch))
    colors = {
                0: np.array((255,0,0)),
                1: np.array((128,128,128)),
                2: np.array((0,255,0))
            }
    pm = lambda x: x.permute(0, 3, 1, 2)
    squash = lambda x: (x - x.min()) / (x.max() - x.min())
    x, y = batch
    y_mask = np.sum(y.cpu().numpy(), axis=3) == 0
    y_hat = frame.act(frame.infer(x))
    y = np.argmax(y.cpu().numpy(), axis=3) + 1
    #y_hat = np.argmax(y_hat.cpu().numpy(), axis=3) + 1
    y_hat = y_hat.cpu().numpy()[:,:,:,1][:,:,:,None]
    y[y_mask] = 0
    #y_hat[y_mask] = 0
    _y = np.zeros((y.shape[0], y.shape[1], y.shape[2], 3))
    #_y_hat = np.zeros((y_hat.shape[0], y_hat.shape[1], y_hat.shape[2], 3))

    for i in range(len(colors)):
        _y[y == i] = colors[i]
        #_y_hat[y_hat == i] = colors[i]
    y = _y
    #y_hat = _y_hat
    x = torch.clamp(x, 0, 1)
    writer.add_image(f"{stage}/x", make_grid(pm(squash(x[:,:,:,[0,1,2]]))), epoch)
    writer.add_image(f"{stage}/y", make_grid(pm(squash(torch.tensor(y)))), epoch)    
    writer.add_image(f"{stage}/y_hat", make_grid(pm(squash(torch.tensor(y_hat)))), epoch)
    
def get_loss(outchannels, opts=None):
    if opts is None:
        return diceloss()
    if opts.weights == "None":
        loss_weight = np.ones(outchannels) / outchannels
    else:
        loss_weight = opts.weights
    if opts.label_smoothing == "None":
        label_smoothing = 0
    else:
        label_smoothing = opts.label_smoothing
    if opts.name == "dice":
        loss_fn = diceloss(act=torch.nn.Softmax(dim=1), w=loss_weight,
                            outchannels=outchannels, label_smoothing=label_smoothing, masked=opts.masked)
    elif opts.name == "iou":
        loss_fn = iouloss(act=torch.nn.Softmax(dim=1), w=loss_weight, 
                            outchannels=outchannels, masked = opts.masked)
    elif opts.name == "ce":
        loss_fn = celoss(act=torch.nn.Softmax(dim=1), w=loss_weight, 
                            outchannels=outchannels, masked = opts.masked)
    elif opts.name == "nll":
        loss_fn = nllloss(act=torch.nn.Softmax(dim=1), w=loss_weight, 
                            outchannels=outchannels, masked = opts.masked)
    elif opts.name == "focal":
        loss_fn = focalloss(act=torch.nn.Softmax(dim=1), w=loss_weight, 
                            outchannels=outchannels, masked = opts.masked)
    elif opts.name == "custom":
        loss_fn = customloss(act=torch.nn.Softmax(dim=1), w=loss_weight, 
                            outchannels=outchannels, masked = opts.masked)                  
    else:
        raise ValueError("Loss must be defined!")
    return loss_fn

def get_metrics(tp, fp, fn, metrics_opts):
    """
    Aggregate --inplace-- the mean of a matrix of tensor (across rows)
    Args:
        metrics (dict(troch.Tensor)): the matrix to get mean of
    """
    metrics = dict.fromkeys(metrics_opts, np.zeros(2))
    for metric, arr in metrics.items():
        metric_fun = globals()[metric]
        metrics[metric] = metric_fun(tp, fp, fn)
    return metrics

def print_conf(conf):
    for key, value in conf.items():
        log(logging.INFO, "{} = {}".format(key, value))


def print_metrics(conf, train_metric, val_metric, round=2):
    train_classes, val_classes = dict(), dict()
    for i, c in enumerate(conf.log_opts.mask_names):
        train_metric_log, val_metric_log = dict(), dict()
        for metric in conf.metrics_opts:
            train_metric_log[metric] = np.round(train_metric[metric][i].item(), 2)
            val_metric_log[metric] = np.round(val_metric[metric][i].item(), 2)
        train_classes[c] = train_metric_log
        val_classes[c] = val_metric_log
    log(logging.INFO, "Train | {}".format(train_classes))
    log(logging.INFO, "Val | {}\n".format(val_classes))
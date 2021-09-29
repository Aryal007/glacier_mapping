#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 23:18:43 2020

@author: mibook

Training/Validation Functions
"""

from torchvision.utils import make_grid
import torch
import numpy as np
from coastal_mapping.model.metrics import diceloss, balancedloss
import pdb
from PIL import Image
from .metrics import *

def train_epoch(loader, frame, metrics_opts, masked, grad_accumulation_steps=None):
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
    n_classes = 2
    loss, batch_loss, tp, fp, fn = 0, 0, torch.zeros(n_classes), torch.zeros(n_classes), torch.zeros(n_classes)

    for i, (x,y) in enumerate(loader):
        y_hat, _loss = frame.optimize(x, y)
        if grad_accumulation_steps != "None":
            if (i+1) % grad_accumulation_steps == 0:
                frame.zero_grad()
        else:
            frame.zero_grad()
        loss += _loss.item()
        y_hat = frame.segment(y_hat)
        _tp, _fp, _fn = frame.metrics(y_hat, y, masked)
        log_batch(i, loss, len(loader), loader.batch_size)
        tp += _tp
        fp += _fp 
        fn += _fn 
    frame.zero_grad()
    
    metrics = get_metrics(tp, fp, fn, metrics_opts)
        
    return loss / len(loader.dataset), metrics


def validate(loader, frame, metrics_opts, masked):
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
    n_classes = 2
    loss, tp, fp, fn = 0, torch.zeros(n_classes), torch.zeros(n_classes), torch.zeros(n_classes)
    channel_first = lambda x: x.permute(0, 3, 1, 2)
    frame.model.eval()
    for x, y in loader:
        with torch.no_grad():
            y_hat = frame.infer(x)
            loss += frame.calc_loss(channel_first(y_hat), channel_first(y)).item()
            y_hat = frame.segment(y_hat)
            _tp, _fp, _fn = frame.metrics(y_hat, y, masked)
            tp += _tp 
            fp += _fp 
            fn += _fn 
    frame.val_operations(loss/len(loader.dataset))
    metrics = get_metrics(tp, fp, fn, metrics_opts)  
    return loss / len(loader.dataset), metrics


def log_batch(i, loss, n_batches, batch_size):
    """Helper to log a training batch

    :param i: Current batch index
    :type i: int
    :param loss: current epoch loss
    :type loss: float
    :param batch_size: training batch size
    :type batch_size: int
    """
    print(
        f"Training batch {i+1} of {n_batches}, Loss = {loss/batch_size/(i+1):.5f}",
        end="\r",
        flush=True,
    )

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
                1: np.array((0,0,0)),
                2: np.array((0,255,0))
            }
    pm = lambda x: x.permute(0, 3, 1, 2)
    squash = lambda x: (x - x.min()) / (x.max() - x.min())
    x, y = batch
    y_mask = np.sum(y.cpu().numpy(), axis=3) == 0
    y_hat = frame.act(frame.infer(x))
    y = np.argmax(y.cpu().numpy(), axis=3) + 1
    y_hat = np.argmax(y_hat.cpu().numpy(), axis=3) + 1
    y[y_mask] = 0
    y_hat[y_mask] = 0

    _y = np.zeros((y.shape[0], y.shape[1], y.shape[2], 3))
    _y_hat = np.zeros((y_hat.shape[0], y_hat.shape[1], y_hat.shape[2], 3))

    for i in range(len(colors)):
        _y[y == i] = colors[i]
        _y_hat[y_hat == i] = colors[i]

    y = _y
    y_hat = _y_hat

    x = x.cpu().numpy()
    x = (x - np.min(x, axis=(0,1))) / (np.max(x, axis=(0,1)) - np.min(x, axis=(0,1)))
    x = torch.from_numpy(x)
    writer.add_image(f"{stage}/x", make_grid(pm(squash(x[:,:,:,[0,1,2]]))), epoch)
    writer.add_image(f"{stage}/y", make_grid(pm(squash(torch.tensor(y)))), epoch)    
    writer.add_image(f"{stage}/y_hat", make_grid(pm(squash(torch.tensor(y_hat)))), epoch)
    
def get_loss(outchannels, opts=None):
    if opts is None:
        return diceloss()
        
    if opts["weights"] == "None":
        loss_weight = np.ones(outchannels) / outchannels
    else:
        loss_weight = opts["weights"]
    label_smoothing = opts["label_smoothing"]
    if opts["name"] == "dice":
        loss_fn = diceloss(act=torch.nn.Softmax(dim=1), w=loss_weight,
                            outchannels=outchannels, label_smoothing=label_smoothing, masked=opts["masked"])
    elif opts["name"] == "balanced":
        loss_fn = balancedloss(act=torch.nn.Softmax(dim=1), w=loss_weight, 
                            outchannels=outchannels, masked = opts["masked"])
    else:
        raise ValueError("Loss name must be dice or balanced!")

    return loss_fn

def get_metrics(tp, fp, fn, metrics_opts):
    """Aggregate --inplace-- the mean of a matrix of tensor (across rows)
       Args:
            metrics (dict(troch.Tensor)): the matrix to get mean of"""
    metrics = dict.fromkeys(metrics_opts, np.zeros(2))

    for metric, arr in metrics.items():
        metric_fun = globals()[metric]
        metrics[metric] = metric_fun(tp, fp, fn)

    return metrics
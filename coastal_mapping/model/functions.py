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
from coastal_mapping.model.metrics import diceloss


def train_epoch(loader, frame, metrics_opts, masked):
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
    loss, metrics = 0, {}
    batch_size = len(loader)

    for i, (x,y) in enumerate(loader):
        y_hat, _loss = frame.optimize(x, y)
        loss += _loss
        log_batch(i, loss, batch_size)

        y_hat = frame.segment(y_hat)
        metrics_ = frame.metrics(y_hat, y, metrics_opts, masked)
        update_metrics(metrics, metrics_)

    mean_metrics = agg_metrics(metrics)
        
    return loss / len(loader.dataset), mean_metrics


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
    loss, metrics = 0, {}
    channel_first = lambda x: x.permute(0, 3, 1, 2)
    frame.model.eval()
    for x, y in loader:
        with torch.no_grad():
            y_hat = frame.infer(x)
            loss += frame.calc_loss(channel_first(y_hat), channel_first(y)).item()

            y_hat = frame.segment(y_hat)
            metrics_ = frame.metrics(y_hat, y, metrics_opts, masked)
            update_metrics(metrics, metrics_)

    mean_metrics = agg_metrics(metrics)
            
    return loss / len(loader.dataset), mean_metrics


def log_batch(i, loss, batch_size):
    """Helper to log a training batch

    :param i: Current batch index
    :type i: int
    :param loss: current epoch loss
    :type loss: float
    :param batch_size: training batch size
    :type batch_size: int
    """
    print(
        f"Training batch {i+1} of {batch_size}, Loss = {loss/(i+1):.5f}",
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
    pm = lambda x: x.permute(0, 3, 1, 2)
    squash = lambda x: (x - x.min()) / (x.max() - x.min())

    x, y = batch

    y_hat = frame.act(frame.infer(x))
    y = torch.flatten(y.permute(0, 1, 3, 2), start_dim=2)
    y_hat = torch.flatten(y_hat.permute(0, 1, 3, 2), start_dim=2)

    if epoch == 0:
        writer.add_image(f"{stage}/x", make_grid(pm(squash(x[:, :, :, :3]))), epoch)
        writer.add_image(f"{stage}/y", make_grid(y.unsqueeze(1)), epoch)
    writer.add_image(f"{stage}/y_hat", make_grid(y_hat.unsqueeze(1)), epoch)
    
def get_dice_loss(outchannels, masked):
    if outchannels > 1:
        loss_weight = np.ones(outchannels) / outchannels
        label_smoothing = 0.2
        loss_fn = diceloss(act=torch.nn.Softmax(dim=1), w=loss_weight,
                               outchannels=outchannels, label_smoothing=label_smoothing, masked=masked)
    else:
        loss_fn = diceloss()
    return loss_fn

def update_metrics(main_metrics, batch_metrics):
    """Append --inplace-- a dict of tensors to another element by element
       Args:
            main_metrics (dict(troch.Tensor)): the matrix to append to
            batch_metrics (dict(troch.Tensor)): the new value to append to main_metrics
    """
    for k, v in batch_metrics.items():
        if k in main_metrics:
            main_metrics[k] = torch.cat((main_metrics[k], v.unsqueeze(-1)), 1)
        else:
            main_metrics[k] = v.unsqueeze(-1)

def agg_metrics(metrics):
    """Aggregate --inplace-- the mean of a matrix of tensor (across rows)
       Args:
            metrics (dict(troch.Tensor)): the matrix to get mean of"""
    for k, v in metrics.items():
        metrics[k] = metrics[k].mean(1)

    return metrics
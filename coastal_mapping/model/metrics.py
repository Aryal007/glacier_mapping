#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 23:09:33 2020

@author: mibook

metrics and losses
"""
import torch
import numpy as np

def precision(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fp = ((pred == label) & (true != label)).sum().item()

    if tp == fp == 0:
        return 0

    return tp / (tp + fp)


def tp_fp_fn(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fp = ((pred == label) & (true != label)).sum().item()
    fn = ((pred != label) & (true == label)).sum().item()

    return tp, fp, fn


def recall(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fn = ((pred != label) & (true == label)).sum().item()

    try:
        return tp / (tp + fn)
    except:
        return 0


def pixel_acc(pred, true):
    return (pred == true).sum().item() / true.numel()


def dice(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fp = ((pred == label) & (true != label)).sum().item()
    fn = ((pred != label) & (true == label)).sum().item()

    try:
        return (2 * tp) / (2 * tp + fp + fn)
    except:
        return 0


def IoU(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fp = ((pred == label) & (true != label)).sum().item()
    fn = ((pred != label) & (true == label)).sum().item()

    try:
        return tp / (tp + fp + fn)
    except:
        return 0

class diceloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = w
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        # CE expects loss to have arg-max channel. Dice expects it to have one-hot
        if len(pred.shape) > len(target.shape):
            target = torch.nn.functional.one_hot(target, num_classes=self.outchannels).permute(0, 3, 1, 2)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        if self.masked:
            mask = torch.sum(target, dim=1) == 1
            pred = self.act(pred).permute(0, 2, 3, 1)
            target = target.permute(0, 2, 3, 1)
            intersection = (pred * target)[mask]
            A_sum = (pred * pred)[mask]
            B_sum = (target * target)[mask]
            intersection = intersection.sum(dim=0)
            A_sum = A_sum.sum(dim=0)
            B_sum = B_sum.sum(dim=0)
        else:
            pred = self.act(pred)
            intersection = (pred * target).sum(dim=[0, 2, 3])
            A_sum = (pred * pred).sum(dim=[0, 2, 3])
            B_sum = (target * target).sum(dim=[0, 2, 3])
            
        union = A_sum + B_sum
        dice = 1 - ((2.0 * intersection + self.smooth) / (union + self.smooth))
        dice = dice * torch.tensor(self.w).to(device=dice.device)

        return dice.sum()

class balancedloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Softmax(), smooth=1.0, w=[1.0], outchannels=1, masked = False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = w
        self.outchannels = outchannels
        self.masked = masked

    def forward(self, pred, target):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred = self.act(pred)

        if self.masked:
            mask = torch.sum(target, dim=1) != 1
            pred = pred.permute(0, 2, 3, 1)
            pred[mask] = torch.zeros(pred.shape[3]).to(device)
            pred = pred.permute(0, 3, 1, 2)

        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        # CE expects loss to have arg-max channel. Dice expects it to have one-hot
        if len(pred.shape) > len(target.shape):
            target = torch.nn.functional.one_hot(target, num_classes=self.outchannels).permute(0, 3, 1, 2)

        pred = torch.argmax(pred, dim=1)
        pred = torch.nn.functional.one_hot(pred, num_classes=self.outchannels).permute(0, 3, 1, 2)
        tp = ((pred == 1) & (target == 1)).sum(dim=(0, 2, 3)).to(device)
        fp = ((pred == 1) & (target != 1)).sum(dim=(0, 2, 3)).to(device)
        fn = ((pred != 1) & (target == 1)).sum(dim=(0, 2, 3)).to(device)
        denominator = (tp+fp+fn).to(device)

        return (1-torch.sum((tp+self.smooth)/(denominator+self.smooth)*torch.from_numpy(self.w).to(device)))

def l1_reg(params, lambda_reg, device):
    """
    Compute l^1 penalty for parameters list
    """
    penalty = torch.tensor(0.0).to(device)
    for param in params:
        penalty += lambda_reg * torch.sum(abs(param))
    return penalty


def l2_reg(params, lambda_reg, device):
    """
    Compute l^2 penalty for parameters list
    """
    penalty = torch.tensor(0.0).to(device)
    for param in params:
        penalty += lambda_reg * torch.norm(param, 2) ** 2
    return penalty
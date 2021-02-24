#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 23:09:33 2020

@author: mibook

metrics and losses
"""
import torch


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

    return (2 * tp) / (2 * tp + fp + fn)


def IoU(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fp = ((pred == label) & (true != label)).sum().item()
    fn = ((pred != label) & (true == label)).sum().item()

    return tp / (tp + fp + fn)


class diceloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = w
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        pred = self.act(pred)
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        # CE expects loss to have arg-max channel. Dice expects it to have one-hot
        if len(pred.shape) > len(target.shape):
            target = torch.nn.functional.one_hot(target, num_classes=self.outchannels).permute(0, 3, 1, 2)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels
        intersection = (pred * target).sum(dim=[0, 2, 3])
        A_sum = (pred * pred).sum(dim=[0, 2, 3])
        B_sum = (target * target).sum(dim=[0, 2, 3])
        union = A_sum + B_sum

        dice = 1 - ((2.0 * intersection + self.smooth) / (union + self.smooth))
        dice = dice * torch.tensor(self.w).to(device=dice.device)

        return dice.sum()
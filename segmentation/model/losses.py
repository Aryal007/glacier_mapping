#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Sep 30 21:42:33 2021

@author: mibook

metrics
"""
import torch
import pdb
from torchvision.ops import sigmoid_focal_loss
import numpy as np
from skimage.filters import gaussian


class diceloss(torch.nn.Module):
    def __init__(
            self,
            act=torch.nn.Sigmoid(),
            smooth=1.0,
            outchannels=1,
            label_smoothing=0,
            masked=False,
            gaussian_blur_sigma=None):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked
        self.gaussian_blur_sigma = gaussian_blur_sigma

    def forward(self, pred, target):
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones(
                (target.size()[0],
                 target.size()[2],
                 target.size()[3]),
                dtype=torch.bool)

        if self.gaussian_blur_sigma != 'None':
            _target = np.zeros_like(target.cpu())
            for i in range(target.shape[0]):
                for j in range(target.shape[1]):
                    _target[i, j, :, :] = gaussian(
                        target[i, j, :, :].cpu(), self.gaussian_blur_sigma)
            target = torch.from_numpy(_target).to(mask.device)

        target = target * (1 - self.label_smoothing) + \
            self.label_smoothing / self.outchannels
        pred = self.act(pred).permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)

        dice = 1 - ((2.0 * (pred * target)[mask].sum(dim=0) + self.smooth) / (
            pred[mask].sum(dim=0) + target[mask].sum(dim=0) + self.smooth))

        return dice


class iouloss(torch.nn.Module):
    def __init__(
            self,
            act=torch.nn.Sigmoid(),
            smooth=1.0,
            outchannels=1,
            label_smoothing=0,
            masked=False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones(
                (target.size()[0],
                 target.size()[2],
                 target.size()[3]),
                dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + \
            self.label_smoothing / self.outchannels

        pred = self.act(pred).permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        union = A_sum + B_sum - intersection
        iou = 1 - ((intersection + self.smooth) / (union + self.smooth))

        return iou.sum()


class celoss(torch.nn.Module):
    def __init__(
            self,
            act=torch.nn.Sigmoid(),
            smooth=1.0,
            outchannels=1,
            label_smoothing=0,
            masked=False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):

        pred = self.act(pred)
        ce = torch.nn.CrossEntropyLoss()(pred, torch.argmax(target, dim=1).long())
        return ce


class nllloss(torch.nn.Module):
    def __init__(
            self,
            act=torch.nn.Sigmoid(),
            smooth=1.0,
            outchannels=1,
            label_smoothing=0,
            masked=False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones(
                (target.size()[0],
                 target.size()[2],
                 target.size()[3]),
                dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + \
            self.label_smoothing / self.outchannels

        pred = self.act(pred)
        nll = torch.nn.NLLLoss(
            weight=self.w.to(
                device=pred.device))(
            pred,
            torch.argmax(
                target,
                dim=1).long())
        return nll


class focalloss(torch.nn.modules.loss._WeightedLoss):
    def __init__(
            self,
            act=torch.nn.Sigmoid(),
            smooth=1.0,
            outchannels=1,
            label_smoothing=0,
            masked=False,
            gamma=2):
        super().__init__()

    def forward(self, pred, target):
        focal_loss = sigmoid_focal_loss(
            pred, target, alpha=-1, gamma=2, reduction="mean")
        return focal_loss


class customloss(torch.nn.modules.loss._WeightedLoss):
    def __init__(
            self,
            act=torch.nn.Sigmoid(),
            smooth=1.0,
            outchannels=1,
            label_smoothing=0,
            masked=False,
            gamma=2):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones(
                (target.size()[0],
                 target.size()[2],
                 target.size()[3]),
                dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + \
            self.label_smoothing / self.outchannels
        _pred = self.act(pred)
        ce = torch.nn.CrossEntropyLoss(
            weight=self.w.to(
                device=_pred.device))(
            _pred, torch.argmax(
                target, dim=1).long())
        pred = self.act(pred).permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)

        intersection = (pred * target)[mask].sum(dim=0)
        union = pred[mask].sum(dim=0) + target[mask].sum(dim=0) - intersection
        iou = 1 - ((intersection + self.smooth) / (union + self.smooth))

        return iou.sum() + ce

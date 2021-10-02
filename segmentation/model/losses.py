#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Sep 30 21:42:33 2021

@author: mibook

metrics
"""
import torch, pdb
from torchvision.ops import sigmoid_focal_loss

class diceloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        pred = self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        total_sum = A_sum + B_sum
        dice = 1 - ((2.0 * intersection + self.smooth) / (total_sum + self.smooth))

        dice = dice * self.w.to(device=dice.device)
        return dice.sum()

class iouloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        pred = self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        union = A_sum + B_sum - intersection
        iou =  -(intersection + self.smooth) / (union + self.smooth)
        iou = iou * self.w.to(device=iou.device)
        return iou.sum()

class celoss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        pred = self.act(pred)
        ce = torch.nn.CrossEntropyLoss(weight=self.w.to(device=pred.device))(pred, torch.argmax(target, dim=1).long())
        return ce

class nllloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        pred = self.act(pred)
        nll = torch.nn.NLLLoss(weight=self.w.to(device=pred.device))(pred, torch.argmax(target, dim=1).long())
        return nll


class senseloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        pred = self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        union = A_sum + B_sum - intersection
        iou =  -(intersection + self.smooth) / (union + self.smooth)
        iou = iou * self.w.to(device=iou.device)
        return iou.sum()


class focalloss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False, gamma=2):
        super().__init__()

    def forward(self, pred, target):
        focal_loss = sigmoid_focal_loss(pred, target, alpha = -1, gamma = 2, reduction = "mean")
        return focal_loss


class customloss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False, gamma=2):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        focal_loss = sigmoid_focal_loss(pred, target, alpha = -1, gamma = 2, reduction = "mean")
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        pred = self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        union = A_sum + B_sum - intersection
        iou =  -(intersection + self.smooth) / (union + self.smooth)
        iou = iou * self.w.to(device=iou.device)

        return iou.sum() + focal_loss

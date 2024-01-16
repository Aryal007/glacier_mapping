#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Sep 30 21:42:33 2021

@author: mibook

metrics
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb
from torchvision.ops import sigmoid_focal_loss
import numpy as np
from skimage.filters import gaussian


class diceloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, outchannels=1, label_smoothing=0, 
            masked=False, boundary=0.5, gaussian_blur_sigma=None):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.boundary_weight = boundary

    def forward(self, pred, target):
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)

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

        dice = dice*torch.tensor([0.95, 0.05]).to(dice.device)

        return dice.sum()

class boundaryloss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, one_hot_gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, C, H, w)
        Return:
            - boundary loss, averaged over mini-batch
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss

class iouloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, outchannels=1, label_smoothing=0, masked=False):
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
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + \
            self.label_smoothing / self.outchannels

        pred = self.act(pred).permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        union = A_sum + B_sum - intersection
        iou = 1 - ((intersection + self.smooth) / (union + self.smooth))
        
        return iou


class celoss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, outchannels=1, label_smoothing=0, masked=False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        pred = self.act(pred)
        ce = torch.nn.CrossEntropyLoss(reduction='none')(pred, torch.argmax(target, dim=1).long())
        return ce


class nllloss(torch.nn.Module):
    def __init__(self,act=torch.nn.Sigmoid(),smooth=1.0,outchannels=1,label_smoothing=0,masked=False):
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
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        focal_loss = sigmoid_focal_loss(
            pred, target, alpha=-1, gamma=3, reduction="mean")
        return focal_loss


class customloss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, outchannels=1,
            label_smoothing=0, masked=True, theta0=3, theta=5):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, target):
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)

        #target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels
        
        n, c, _, _ = pred.shape
        # softmax so that predicted map can be distributed in [0, 1]
        pred = self.act(pred)
        # boundary map
        gt_b = F.max_pool2d(1 - target, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - target
        pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred
        # extended boundary map
        gt_b_ext = F.max_pool2d(gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)
        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)
        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)
        # summing BF1 Score for each class and average over mini-batch
        boundaryloss = torch.mean(1 - BF1)

        pred = pred.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        diceloss = 1 - ((2.0 * (pred * target)[mask].sum(dim=0) + self.smooth) / (pred[mask].sum(dim=0) + target[mask].sum(dim=0) + self.smooth))
        diceloss = diceloss*torch.tensor([0.0, 1.0]).to(diceloss.device)
        
        return diceloss, boundaryloss

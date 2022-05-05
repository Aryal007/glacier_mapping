#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Sep 30 21:42:33 2021

@author: mibook

metrics
"""
import torch
import torch.nn.functional as F
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
        
        boundary_target, boundary_pred = self.boundary(pred, target)

        target = target * (1 - self.label_smoothing) + \
            self.label_smoothing / self.outchannels
        
        pred = self.act(pred).permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        
        dice = 1 - ((2.0 * (pred * target)[mask].sum(dim=0) + self.smooth) / (
            pred[mask].sum(dim=0) + target[mask].sum(dim=0) + self.smooth))
        #dice_boundary = 1 - ((2.0 * (boundary_pred * boundary_target)[mask].sum(dim=0) + self.smooth) / (
        #    boundary_pred[mask].sum(dim=0) + boundary_target[mask].sum(dim=0) + self.smooth))
        #dice_boundary = dice_boundary[1]
        #ce_boundary = torch.nn.CrossEntropyLoss()(pred, torch.argmax(target, dim=1).long())/6

        return dice

    def boundary(self, gen_frames, gt_frames):
        def gradient(x):
            # idea from tf.image.image_gradients(image)
            # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
            # x: (b,c,h,w), float32 or float64
            # dx, dy: (b,c,h,w)

            h_x = x.size()[-2]
            w_x = x.size()[-1]
            # gradient step=1
            left = x
            right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
            top = x
            bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

            # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
            dx, dy = right - left, bottom - top 
            # dx will always have zeros in the last column, right-left
            # dy will always have zeros in the last row,    bottom-top
            dx[:, :, :, -1] = 0
            dy[:, :, -1, :] = 0

            return dx, dy

        # gradient
        gen_dx, gen_dy = gradient(gen_frames)
        gt_dx, gt_dy = gradient(gt_frames)

        boundary_gt = torch.sqrt(gt_dx**2+gt_dy**2)
        boundary_gen = torch.sqrt(gen_dx**2+gen_dy**2)

        return boundary_gt.permute(0, 2, 3, 1), self.act(boundary_gen).permute(0, 2, 3, 1)

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
            pred, target, alpha=-1, gamma=3, reduction="mean")
        return focal_loss


class customloss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, outchannels=1,
            label_smoothing=0, masked=False, gamma=2):
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
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels
        pred = self.act(pred).permute(0, 2, 3, 1)
        ce = torch.nn.CrossEntropyLoss(ignore_index=0)(pred[mask], torch.argmax(target, dim=1).long()[mask])
        target = target.permute(0, 2, 3, 1)
        intersection = (pred * target)[mask].sum(dim=0)
        union = pred[mask].sum(dim=0) + target[mask].sum(dim=0) - intersection
        iou = 1 - ((intersection + self.smooth) / (union + self.smooth))

        return iou + ce/2

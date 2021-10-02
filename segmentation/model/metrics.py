#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 23:09:33 2020

@author: mibook

metrics and regularizations
"""
import numpy as np
import torch, pdb

def precision(tp, fp, fn):
    try:
        return tp / (tp + fp)
    except:
        return 0


def tp_fp_fn(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fp = ((pred == label) & (true != label)).sum().item()
    fn = ((pred != label) & (true == label)).sum().item()

    return tp, fp, fn


def recall(tp, fp, fn):
    try:
        return tp / (tp + fn)
    except:
        return 0


def dice(tp, fp, fn):
    try:
        return (2 * tp) / (2 * tp + fp + fn)
    except:
        return 0


def IoU(tp, fp, fn):
    try:
        return tp / (tp + fp + fn)
    except:
        return 0


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
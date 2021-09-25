#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 22:59:16 2020

@author: mibook

Frame to Combine Model with Optimizer

This wraps the model and optimizer objects needed in training, so that each
training step can be concisely called with a single method.
"""
from pathlib import Path
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
import os
from .metrics import *
from .unet import *

class Framework:
    """
    Class to Wrap all the Training Steps

    """

    def __init__(self, loss_fn, model_opts=None, optimizer_opts=None,
                 reg_opts=None, device=None):
        """
        Set Class Attrributes
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if optimizer_opts is None:
            optimizer_opts = {"name": "Adam", "args": {"lr": 0.001}}
        self.multi_class = True if model_opts.args.outchannels > 1 else False
        self.num_classes = model_opts.args.outchannels    
        self.loss_fn = loss_fn.to(self.device)
        self.model = Unet(**model_opts.args).to(self.device)
        optimizer_def = getattr(torch.optim, optimizer_opts["name"])
        self.optimizer = optimizer_def(self.model.parameters(), **optimizer_opts["args"])
        self.lrscheduler = ReduceLROnPlateau(self.optimizer, "min",
                                             verbose = True, 
                                             patience=5,
                                             factor = 0.5,
                                             min_lr = 1e-9)
        self.lrscheduler2 = ExponentialLR(self.optimizer, 0.99, verbose=True)
        self.reg_opts = reg_opts


    def optimize(self, x, y):
        """
        Take a single gradient step

        Args:
            X: raw training data
            y: labels
        Return:
            optimization
        """
        x = x.permute(0, 3, 1, 2).to(self.device)
        y = y.permute(0, 3, 1, 2).to(self.device)

        y_hat = self.model(x)
        
        loss = self.calc_loss(y_hat, y)
        loss.backward()
        return y_hat.permute(0, 2, 3, 1), loss
    
    def zero_grad(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


    def val_operations(self, val_loss):
        """
        Update the LR Scheduler
        """
        #self.lrscheduler2.step()
        self.lrscheduler.step(val_loss)

    def save(self, out_dir, epoch):
        """
        Save a model checkpoint
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model_path = Path(out_dir, f"model_{epoch}.pt")
        optim_path = Path(out_dir, f"optim_{epoch}.pt")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optim_path)

    def infer(self, x):
        """ Make a prediction for a given x

        Args:
            x: input x

        Return:
            Prediction

        """
        x = x.permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            return self.model(x).permute(0, 2, 3, 1)

    def calc_loss(self, y_hat, y):
        """ Compute loss given a prediction

        Args:
            y_hat: Prediction
            y: Label

        Return:
            Loss values

        """
        y_hat = y_hat.to(self.device)
        y = y.to(self.device)
        loss = self.loss_fn(y_hat, y)
        if self.reg_opts:
            for reg_type in self.reg_opts.keys():
                reg_fun = globals()[reg_type]
                penalty = reg_fun(
                    self.model.parameters(),
                    self.reg_opts[reg_type],
                    self.device
                )
                loss += penalty

        return loss


    def metrics(self, y_hat, y, masked):
        """ Loop over metrics in train.yaml

        Args:
            y_hat: Predictions
            y: Labels

        Return:
            results

        """
        y_hat = y_hat.to(self.device)
        y = y.to(self.device)
        n_classes = y.shape[3]

        if masked:
            mask = torch.sum(y, dim=3) == 0

        y_hat = np.argmax(y_hat.cpu().numpy(), axis=3)+1
        y = np.argmax(y.cpu().numpy(), axis=3)+1

        if masked:
            y_hat[mask] = 0
            y[mask] = 0
        
        tp, fp, fn = torch.zeros(n_classes), torch.zeros(n_classes), torch.zeros(n_classes)
        for i in range(n_classes):
            _y_hat = (y_hat == i+1).astype(np.uint8)
            _y = (y == i+1).astype(np.uint8)
            _tp, _fp, _fn = tp_fp_fn(_y_hat, _y)
            tp[i] = _tp
            fp[i] = _fp
            fn[i] = _fn
            
        return tp, fp, fn
    
    def segment(self, y_hat):
        """Predict a class given logits
        Args:
            y_hat: logits output
        Return:
            Probability of class in case of binary classification
            or one-hot tensor in case of multi class"""
        if self.multi_class:
            y_hat = torch.argmax(y_hat, axis=3)
            y_hat = torch.nn.functional.one_hot(y_hat, num_classes=self.num_classes)
        else:
            y_hat = torch.sigmoid(y_hat)
            
        return y_hat
    
    def act(self, logits):
        """Applies activation function based on the model
        Args:
            y_hat: logits output
        Returns:
            logits after applying activation function"""

        if self.multi_class:
            y_hat = torch.nn.Softmax(3)(logits)
        else:
            y_hat = torch.sigmoid(logits)
        return y_hat

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def optim_load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def load_best(self, model_path):
        print(f"Validation loss higher than previous for 3 steps, loading previous state")
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        self.load_state_dict(state_dict)
    
    def save_best(self, out_dir):
        print(f"Current validation loss lower than previous, saving current state")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model_path = Path(out_dir, f"model_best.h5")
        optim_path = Path(out_dir, f"optim_best.h5")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optim_path)
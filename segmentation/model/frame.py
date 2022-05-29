#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 22:59:16 2020

@author: mibook

Frame to Combine Model with Optimizer

This wraps the model and optimizer objects needed in training, so that each
training step can be concisely called with a single method.
"""
from .unet import *
from .metrics import *

import numpy as np
import os, torch, math, random
import pdb
from tqdm import tqdm
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

class Framework:
    """
    Class to Wrap all the Training Steps

    """

    def __init__(self, loss_fn, model_opts=None, optimizer_opts=None,
                 reg_opts=None, loss_opts=None, device=None):
        """
        Set Class Attrributes
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(device)
        if optimizer_opts is None:
            optimizer_opts = {"name": "Adam", "args": {"lr": 0.001}}
        self.multi_class = True if model_opts.args.outchannels > 1 else False
        self.num_classes = model_opts.args.outchannels
        if loss_opts is not None:
            self.loss_alpha = torch.tensor([loss_opts.alpha]).to(self.device)
        else:
            self.loss_alpha = torch.tensor([0.0]).to(self.device)
        self.k = 1
        self.sigma1, self.sigma2 = torch.tensor([1.0]).to(self.device), torch.tensor([1.0]).to(self.device)
        self.sigma1, self.sigma2 = self.sigma1.requires_grad_(), self.sigma2.requires_grad_()
        #self.loss_alpha = self.loss_alpha.requires_grad_()
        self.loss_fn = loss_fn.to(self.device)
        self.model = Unet(**model_opts.args).to(self.device)
        _optimizer_params = [{'params': self.model.parameters(), **optimizer_opts["args"]},
                            {'params': self.sigma1, **optimizer_opts["args"]},
                            {'params': self.sigma2, **optimizer_opts["args"]}]
        if loss_opts is None:
            self.loss_weights = torch.tensor([1.0, 1.0, 1.0]).to(self.device)
        else:
            self.loss_weights = torch.tensor(loss_opts.weights).to(self.device)
        optimizer_def = getattr(torch.optim, optimizer_opts["name"])
        self.optimizer = optimizer_def(_optimizer_params)
        self.lrscheduler = ReduceLROnPlateau(self.optimizer, "min",
                                             verbose=True,
                                             patience=15,
                                             factor=0.1,
                                             min_lr=1e-9)
        #self.lrscheduler2 = CyclicLR(
        #    self.optimizer,
        #    base_lr=optimizer_opts["args"]["lr"] * 0.001,
        #    max_lr=optimizer_opts["args"]["lr"],
        #    mode='triangular2',
        #    step_size_up=30,
        #    verbose=True,
        #    cycle_momentum=False)
        #self.lrscheduler2 = ExponentialLR(self.optimizer, 0.795, verbose=True)
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
        self.optimizer.zero_grad()

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def step(self):
        self.optimizer.step()

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
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved model {epoch}")

    def infer(self, x):
        """ Make a prediction for a given x

        Args:
            x: input x

        Return:
            Prediction

        """
        x = x.permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            y = self.model(x)
        return y.permute(0, 2, 3, 1)

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
        #loss = self.loss_fn(y_hat, y)
        diceloss, boundaryloss = self.loss_fn(y_hat, y)
        diceloss = diceloss.sum() * self.k
        #loss = torch.add(torch.mul(diceloss.clone(), self.loss_alpha), torch.mul(boundaryloss.clone(), (1-self.loss_alpha)))
        loss = torch.add(
                torch.add(
                    torch.mul(torch.div(1, torch.mul(2, torch.square(self.sigma1))), diceloss.clone()), 
                    torch.mul(torch.div(1, torch.mul(2, torch.square(self.sigma2))), boundaryloss.clone())
                    ),
                torch.abs(torch.log(torch.mul(self.sigma1, self.sigma2)))
                )
        if self.reg_opts:
            for reg_type in self.reg_opts.keys():
                reg_fun = globals()[reg_type]
                penalty = reg_fun(self.model.parameters(),self.reg_opts[reg_type],self.device)
                loss += penalty
        return loss.abs()
    
    def get_loss_alpha(self):
        #return self.loss_alpha.item()
        return (self.sigma1.item(), self.sigma2.item())

    def metrics(self, y_hat, y, mask, threshold):
        """ Loop over metrics in train.yaml

        Args:
            y_hat: Predictions
            y: Labels

        Return:
            results

        """
        n_classes = y.shape[3]
        _y_hat = np.zeros((y_hat.shape[0], y_hat.shape[1], y_hat.shape[2]))
        y_hat = y_hat.detach().cpu().numpy()
        for i in range(1, n_classes):
            _y_hat[y_hat[:, :, :, i] >= threshold[i - 1]] = i + 1
        _y_hat[_y_hat == 0] = 1
        _y_hat[mask] = -1
        y_hat = _y_hat

        y = np.argmax(y.cpu().numpy(), axis=3) + 1
        y[mask] = -1

        tp, fp, fn = torch.zeros(n_classes), torch.zeros(
            n_classes), torch.zeros(n_classes)
        for i in range(0, n_classes):
            _y_hat = (y_hat == i + 1).astype(np.uint8)
            _y = (y == i + 1).astype(np.uint8)
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
            y_hat = torch.nn.functional.one_hot(
                y_hat, num_classes=self.num_classes)
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
        torch.save(self.model.state_dict(), model_path)

    def freeze_layers(self, layers=None):
        for i, layer in enumerate(self.model.parameters()):
            if layers is None:
                layer.requires_grad = False
            elif i < layers:  # Freeze 60 out of 75 layers, retrain on last 15 only
                layer.requires_grad = False
            else:
                pass

    def find_lr(self, train_loader, init_value, final_value):
        number_in_epoch = len(train_loader) - 1
        update_step = (final_value / init_value) ** (1 / number_in_epoch)
        lr = init_value
        self.optimizer.param_groups[0]["lr"] = lr
        best_loss = 0.0
        batch_num = 0
        losses = []
        log_lrs = []
        iterator = tqdm(
            train_loader,
            desc="Current lr=XX.XX Steps=XX Loss=XX.XX Best lr=XX.XX ")
        for i, data in enumerate(iterator):
            batch_num += 1
            inputs, labels = data
            self.optimizer.zero_grad()
            inputs = inputs.permute(0, 3, 1, 2).to(self.device)
            labels = labels.permute(0, 3, 1, 2).to(self.device)
            outputs = self.model(inputs)
            loss = self.calc_loss(outputs, labels)
            # Crash out if loss explodes
            if batch_num > 1 and loss > 4 * best_loss:
                return log_lrs[10:-5], losses[10:-5]
            # Record the best loss
            if loss < best_loss or batch_num == 1:
                best_loss = loss
                best_lr = lr
            # Do the backward pass and optimize
            loss.backward()
            self.optimizer.step()
            iterator.set_description(
                "Current lr=%5.9f Steps=%d Loss=%5.3f Best lr=%5.9f " %
                (lr, i, loss, best_lr))
            # Store the values
            losses.append(loss.detach())
            log_lrs.append(math.log10(lr))
            # Update the lr for the next step and store
            lr = lr * update_step
            self.optimizer.param_groups[0]["lr"] = lr
        return log_lrs[10:-5], losses[10:-5]

    def get_model_device(self):
        return self.model, self.device
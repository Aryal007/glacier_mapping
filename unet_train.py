#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:59:22 2021

@author: Aryal007
"""
from segmentation.data.data import fetch_loaders
from segmentation.model.frame import Framework
import segmentation.model.functions as fn

import yaml
import json
import pathlib
import warnings
import logging
import gc
import torch
import random
import pdb
from torch.utils.tensorboard import SummaryWriter
from addict import Dict
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    conf = Dict(yaml.safe_load(open('./conf/unet_train.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    class_name = conf.class_name
    run_name = conf.run_name
    processed_dir = data_dir
    train_loader, val_loader = fetch_loaders(
        processed_dir, conf.batch_size, conf.use_channels, conf.normalize, val_folder='val')
    loss_fn = fn.get_loss(conf.model_opts.args.outchannels, conf.loss_opts)
    frame = Framework(
        loss_fn=loss_fn,
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts,
        loss_opts=conf.loss_opts,
        device=int(conf.gpu_rank)
    )

    if conf.fine_tune:
        fn.log(logging.INFO, f"Finetuning the model")
        run_name = conf.run_name + "_finetuned"
        model_path = f"{data_dir}/runs/{conf.run_name}/models/model_final.pt"
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        frame.load_state_dict(state_dict)
        frame.freeze_layers()

    if conf.find_lr:
        fn.find_lr(frame, train_loader, init_value=1e-9, final_value=10)
        exit()

    # Setup logging
    writer = SummaryWriter(f"{data_dir}/runs/{run_name}/logs/")
    writer.add_text("Configuration Parameters", json.dumps(conf))
    out_dir = f"{data_dir}/runs/{run_name}/models/"
    loss_val = np.inf

    fn.print_conf(conf)

    with open(f"{data_dir}/runs/{run_name}/conf.json", 'w') as f:
        j = json.dumps(conf, sort_keys=True)
        f.write(j)
    
    fn.log(
        logging.INFO,
        "# Training Instances = {}, # Validation Instances = {}".format(
            len(train_loader),
            len(val_loader)))

    _normalize = np.load(data_dir / "normalize_train.npy")
    if conf.normalize == "min-max":
        _normalize = (_normalize[2][conf.use_channels],
                      _normalize[3][conf.use_channels])
    elif conf.normalize == "mean-std":
        _normalize = (_normalize[0][conf.use_channels],
                      _normalize[1][conf.use_channels])
    else:
        raise ValueError("Normalize must be min-max or mean-std")

    for epoch in range(1, conf.epochs + 1):
        # train loop
        loss_train, train_metric, loss_weights = fn.train_epoch(epoch, train_loader, frame, conf)
        fn.log_metrics(writer, train_metric, epoch, "train", conf.log_opts.mask_names)

        # validation loop
        new_loss_val, val_metric = fn.validate(epoch, val_loader, frame, conf)
        fn.log_metrics(writer, val_metric, epoch, "val", conf.log_opts.mask_names)

        if (epoch - 1) % 5 == 0:
            fn.log_images( writer, frame, train_loader, epoch, "train", conf.threshold, conf.normalize, _normalize)
            fn.log_images(writer, frame, val_loader, epoch, "val", conf.threshold, conf.normalize, _normalize)

        # Save best model
        if new_loss_val < loss_val:
            frame.save(out_dir, "best")
            loss_val = float(new_loss_val)

        lr = fn.get_current_lr(frame)
        writer.add_scalars("Loss", {"train": loss_train, "val": new_loss_val}, epoch)

        writer.add_scalar("lr", lr, epoch)

        fn.print_metrics(conf, train_metric, val_metric)
        torch.cuda.empty_cache()
        writer.flush()
        gc.collect()

    frame.save(out_dir, "final")
    writer.close()

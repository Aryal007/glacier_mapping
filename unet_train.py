#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:59:22 2021

@author: mibook
"""
from coastal_mapping.data.data import fetch_loaders
from coastal_mapping.model.frame import Framework
import coastal_mapping.model.functions as fn

from torch.utils.tensorboard import SummaryWriter
import yaml, json, pathlib
from addict import Dict

conf = Dict(yaml.safe_load(open('./conf/train.yaml')))
data_dir = data_dir = pathlib.Path(conf.data_dir)
processed_dir = data_dir / "processed"

loaders = fetch_loaders(processed_dir, conf.batch_size)

if conf.loss_type == "dice":
    loss_fn = fn.get_dice_loss(conf.model_opts.args.outchannels, conf.loss_masked)    
else:
    loss_fn = None
    
frame = Framework(
    loss_fn = loss_fn,
    model_opts=conf.model_opts,
    optimizer_opts=conf.optim_opts,
    reg_opts=conf.reg_opts
)

# Setup logging
writer = SummaryWriter(f"{data_dir}/runs/{conf.run_name}/logs/")
writer.add_text("Configuration Parameters", json.dumps(conf))
out_dir = f"{data_dir}/runs/{conf.run_name}/models/"

train_batch = next(iter(loaders["train"]))
val_batch = next(iter(loaders["val"]))

for epoch in range(conf.epochs):
    # train loop
    loss = {}
    loss["train"], train_metric = fn.train_epoch(loaders["train"], frame, conf.metrics_opts)
    fn.log_metrics(writer, train_metric, epoch+1, "train", conf.log_opts.mask_names)
    fn.log_images(writer, frame, train_batch, epoch, "train")

    # validation loop
    loss["val"], val_metric = fn.validate(loaders["val"], frame, conf.metrics_opts)
    fn.log_metrics(writer, val_metric, epoch+1, "val", conf.log_opts.mask_names)
    fn.log_images(writer, frame, val_batch, epoch, "val")

    writer.add_scalars("Loss", loss, epoch)
    # Save model
    if epoch % conf.save_every == 0:
        frame.save(out_dir, epoch)

    print(f"{epoch+1}/{conf.epochs} | train: {loss['train']} | val: {loss['val']}")

frame.save(out_dir, "final")
writer.close()
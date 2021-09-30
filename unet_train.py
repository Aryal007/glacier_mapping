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
import warnings, pdb
import torch, time
import numpy as np
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    conf = Dict(yaml.safe_load(open('./conf/unet_train.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    class_name = conf.class_name
    run_name = conf.run_name
    processed_dir = data_dir / "processed"
    loaders = fetch_loaders(processed_dir, conf.batch_size, conf.use_channels)
    loss_fn = fn.get_loss(conf.model_opts.args.outchannels, conf.loss_opts)            
    frame = Framework(
        loss_fn = loss_fn,
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts
    )

    if conf.fine_tune:
        print(f"Finetuning the model")
        run_name = conf.run_name+"_finetuned"
        model_path = f"{data_dir}/runs/{conf.run_name}/models/model_final.pt"
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        frame.load_state_dict(state_dict)
        frame.freeze_layers()

    # Setup logging
    writer = SummaryWriter(f"{data_dir}/runs/{run_name}/logs/")
    writer.add_text("Configuration Parameters", json.dumps(conf))
    out_dir = f"{data_dir}/runs/{run_name}/models/"
    val_loss = np.inf

    for epoch in range(conf.epochs):
        # train loop
        loss = {}
        start = time.time()
        loss["train"], train_metric = fn.train_epoch(loaders["train"], frame, conf.metrics_opts, conf.loss_masked, conf.grad_accumulation_steps)
        fn.log_metrics(writer, train_metric, epoch+1, "train", conf.log_opts.mask_names)
        train_time = time.time() - start

        # validation loop
        start = time.time()
        loss["val"], val_metric = fn.validate(loaders["val"], frame, conf.metrics_opts, conf.loss_masked)
        fn.log_metrics(writer, val_metric, epoch+1, "val", conf.log_opts.mask_names)
        val_time = time.time() - start

        if epoch % 5 == 0:
            fn.log_images(writer, frame, loaders["train"], epoch, "train")
            fn.log_images(writer, frame, loaders["val"], epoch, "val")

        writer.add_scalars("Loss", loss, epoch)
        # Save model
        if epoch % conf.save_every == 0:
            frame.save(out_dir, epoch)
        if conf.epochs - epoch <= 3:
            frame.save(out_dir, epoch)

        print(f"{epoch+1}/{conf.epochs} | train_loss: {loss['train']:.5f} | val_loss: {loss['val']:.5f} \
                | iou: {val_metric['IoU'][0]:.3f}, {val_metric['IoU'][1]:.3f} | precision: {val_metric['precision'][0]:.3f}, {val_metric['precision'][1]:.3f} | recall: {val_metric['recall'][0]:.3f}, {val_metric['recall'][1]:.3f} \
                | train_batch_time: {train_time:.2f} | val_batch_time: {val_time:.2f}")

        writer.flush()

    frame.save(out_dir, "final")
    writer.close()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:59:22 2021

@author: Aryal007
"""
from segmentation.data.data import fetch_loaders
from segmentation.model.frame import Framework
import segmentation.model.functions as fn

import yaml, json, pathlib, warnings, pdb, torch, logging, time
from torch.utils.tensorboard import SummaryWriter
from addict import Dict
import numpy as np

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    conf = Dict(yaml.safe_load(open('./conf/unet_train.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    class_name = conf.class_name
    run_name = conf.run_name
    #processed_dir = data_dir / "processed"
    processed_dir = data_dir
    loaders = fetch_loaders(processed_dir, conf.batch_size, conf.use_channels, val_folder = 'valid')
    loss_fn = fn.get_loss(conf.model_opts.args.outchannels, conf.loss_opts)            
    frame = Framework(
        loss_fn = loss_fn,
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts
    )

    if conf.fine_tune:
        fn.log(logging.INFO, f"Finetuning the model")
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
    #writer.add_graph(frame.get_model(), next(iter(loaders['train'])))
    out_dir = f"{data_dir}/runs/{run_name}/models/"
    val_loss = np.inf
    
    fn.print_conf(conf)
    fn.log(logging.INFO, "# Training Instances = {}, # Validation Instances = {}".format(len(loaders["train"]), len(loaders["val"])))

    for epoch in range(conf.epochs):
        # train loop
        loss = {}
        start = time.time()
        loss["train"], train_metric = fn.train_epoch(epoch, loaders["train"], frame, conf)
        fn.log_metrics(writer, train_metric, epoch+1, "train", conf.log_opts.mask_names)
        train_time = time.time() - start

        # validation loop
        start = time.time()
        loss["val"], val_metric = fn.validate(epoch, loaders["val"], frame, conf)
        fn.log_metrics(writer, val_metric, epoch+1, "val", conf.log_opts.mask_names)
        val_time = time.time() - start

        if epoch % 5 == 0:
            fn.log_images(writer, frame, loaders["train"], epoch, "train")
            fn.log_images(writer, frame, loaders["val"], epoch, "val")

        writer.add_scalars("Loss", loss, epoch)
        # Save model
        # if epoch % conf.save_every == 0:
        #     frame.save(out_dir, epoch)
        # if conf.epochs - epoch <= 3:
        #     frame.save(out_dir, epoch)

        fn.print_metrics(conf, train_metric, val_metric)
        writer.flush()

    frame.save(out_dir, "final")
    writer.close()
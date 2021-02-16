#!/usr/bin/env python
"""
Training/Validation Module

The overall training and validation pipeline has the following structure,

* Initialize loaders (train & validation)
* Initialize the framework
* Train Loop args.e epochs
* Log Epoch level train loss, test loss, metrices, image prediction each s step.
* Save checkpoints after save_every epochs
* models are saved in path/models/name_of_the_run
* tensorboard is saved in path/runs/name_of_the_run
"""
#from glacier_mapping.data.data import fetch_loaders
from .utils.frame import Framework
import utils.functions as fn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

data_dir = pathlib.Path("/content")
conf = Dict(yaml.safe_load(open("conf/train.yaml", "r")))
processed_dir = data_dir / "processed"

args = Dict({
    "batch_size": 16,
    "run_name": "demo",
    "epochs": 151,
    "save_every": 50
})

loaders = fetch_loaders(processed_dir, args.batch_size)
frame = Framework(
    model_opts=conf.model_opts,
    optimizer_opts=conf.optim_opts,
    reg_opts=conf.reg_opts
)

# Setup logging
writer = SummaryWriter(f"{data_dir}/runs/{args.run_name}/logs/")
writer.add_text("Arguments", json.dumps(vars(args)))
writer.add_text("Configuration Parameters", json.dumps(conf))
out_dir = f"{data_dir}/runs/{args.run_name}/models/"

for epoch in range(args.epochs):

    # train loop
    loss_d = {}
    loss_d["train"], metrics = tr.train_epoch(loaders["train"], frame, conf.metrics_opts)
    tr.log_metrics(writer, metrics, loss_d["train"], epoch)
    tr.log_images(writer, frame, next(iter(loaders["train"])), epoch)

    # validation loop
    loss_d["val"], metrics = tr.validate(loaders["val"], frame, conf.metrics_opts)
    tr.log_metrics(writer, metrics, loss_d["val"], epoch, "val")
    tr.log_images(writer, frame, next(iter(loaders["val"])), epoch, "val")

    # Save model
    writer.add_scalars("Loss", loss_d, epoch)
    if epoch % args.save_every == 0:
        frame.save(out_dir, epoch)

    print(f"{epoch}/{args.epochs} | train: {loss_d['train']} | val: {loss_d['val']}")

frame.save(out_dir, "final")
writer.close()
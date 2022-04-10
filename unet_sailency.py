from segmentation.model.frame import Framework
import segmentation.model.functions as fn

import yaml, pdb, os, pathlib, torch
from addict import Dict
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

if __name__ == "__main__":
    labels_dict = {"Clean Ice": 1, "Debris": 2}
    conf = Dict(yaml.safe_load(open('./conf/unet_sailency.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    sailency_dir = data_dir / conf.processed_dir / "sailency" / conf.run_name
    if not os.path.exists(sailency_dir):
        os.makedirs(sailency_dir)
    model_path = data_dir / conf.processed_dir / conf.folder_name / \
        conf.run_name / "models" / "model_best.pt"
    loss_fn = fn.get_loss(conf.model_opts.args.outchannels)
    frame = Framework(
        loss_fn=loss_fn,
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        device=(int(conf.gpu_rank))
    )
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")
    frame.load_state_dict(state_dict)
    model, device = frame.get_model_device()

    arr = np.load(data_dir / conf.processed_dir / "normalize_train.npy")
    if conf.normalize == "mean-std":
        _mean, _std = arr[0][conf.use_channels], arr[1][conf.use_channels]
    if conf.normalize == "min-max":
        _min, _max = arr[2][conf.use_channels], arr[3][conf.use_channels]

    files = sorted(os.listdir(data_dir / conf.processed_dir / conf.split))
    inputs = [x for x in files if "tiff" in x]

    inputs_dict = dict(enumerate(inputs))
    pprint(inputs_dict)
    print("Enter the index of the file you want to use: ")
    n = int(input())
    n = 332
    x_fname = inputs_dict[n]
    print(f"Filename: {x_fname}")

    x = np.load(data_dir / conf.processed_dir / conf.split / x_fname)[:,:,conf.use_channels]
    plt.imshow(x[:,:,[4,2,1]]/255)
    plt.savefig("sailency_x_true.png")
    mask = np.sum(x[:,:,:5], axis=2) == 0
    if conf.normalize == "mean-std":
        x = (x - _mean) / _std
    if conf.normalize == "min-max":
        x = (x - _min) / (_max - _min)
    
    y_fname = x_fname.replace("tiff", "mask")
    y_true = np.load(data_dir / conf.processed_dir / conf.split / y_fname) + 1
    y_true[mask] = 0
    plt.imshow(y_true)
    plt.savefig("sailency_y_true.png")

    x = torch.from_numpy(np.expand_dims(x, axis=0)).float()
    
    x = x.permute(0, 3, 1, 2).to(device)
    y = model(x)
    y = y.permute(0, 2, 3, 1)

    pdb.set_trace()


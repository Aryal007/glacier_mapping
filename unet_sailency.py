from segmentation.model.frame import Framework
import segmentation.model.functions as fn

import yaml, pdb, os, pathlib, torch
from addict import Dict
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

if __name__ == "__main__":
    channels = ["B1", "B2", "B3", "B4", "B5", "B6_VCID1", "B6_VCID2", "B7", "elevation", 
        "slope * sin(aspect)", "slope * cos(aspect)", "curvature", "latitude", "longitude",
        "NDVI", "NDWI", "NDSI", "Hue", "Saturation", "Value"]
    conf = Dict(yaml.safe_load(open('./conf/unet_sailency.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    sailency_dir = data_dir / conf.processed_dir / "sailency" / conf.run_name
    if not os.path.exists(sailency_dir):
        os.makedirs(sailency_dir)
    model_path = data_dir / conf.processed_dir / conf.folder_name / \
        conf.run_name / "models" / "model_best.pt"
    loss_fn = fn.get_loss(conf.model_opts.args.outchannels, conf.loss_opts)
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
    
    '''
    sums = []
    for x_fname in inputs:
        x = np.load(data_dir / conf.processed_dir / conf.split / x_fname)[:,:,conf.use_channels]
        mask = np.sum(x[:,:,:7], axis=2) == 0
        if conf.normalize == "mean-std":
            x = (x - _mean) / _std
        if conf.normalize == "min-max":
            x = (x - _min) / (_max - _min)
        y_fname = x_fname.replace("tiff", "mask")
        y_true = np.load(data_dir / conf.processed_dir / conf.split / y_fname) + 1
        y_true[mask] = 0
        x = torch.from_numpy(np.expand_dims(x, axis=0)).float()
        x = x.permute(0, 3, 1, 2).to(device)
        x = x.requires_grad_()
        y = model(x)
        y = y.permute(0, 2, 3, 1)
        y = torch.nn.Softmax(3)(y)
        y[:,:,:,1].mean().backward()
        _x = x.grad.data.abs().detach().cpu().numpy()[0].transpose(1,2,0)
        _sum =  np.sum(_x, axis=(0,1))
        sums.append(_sum)
    sums = np.asarray(sums)
    scores = np.mean(sums, axis=0)
    scores = dict(zip(channels, np.mean(sums, axis=0)))
    scores = dict(sorted(scores.items(), key=lambda item: item[1]))
    print(scores)
    pdb.set_trace()
    '''
    
    #_y = y
    #plt.imshow(_y.detach().cpu().numpy()[0,:,:,1])
    #plt.savefig("./sailencymap/sailency_y_pred.png")
    #y[:,:,:,1].mean().backward()
    #_x = x.grad.data.abs().detach().cpu().numpy()[0].transpose(1,2,0)

    inputs_dict = dict(enumerate(inputs))
    pprint(inputs_dict)
    print("Enter the index of the file you want to use: ")
    n = int(input())
    n = 13
    x_fname = inputs_dict[n]
    print(f"Filename: {x_fname}")

    fig = plt.figure(figsize=(14, 8))
    grid = plt.GridSpec(4, 7)
    x_plot = fig.add_subplot(grid[0:2, 0:2])
    y_plot = fig.add_subplot(grid[2:4, 0:2])
    b0_plot = fig.add_subplot(grid[0, 2])
    b1_plot = fig.add_subplot(grid[0, 3])
    b2_plot = fig.add_subplot(grid[0, 4])
    b3_plot = fig.add_subplot(grid[0, 5])
    b4_plot = fig.add_subplot(grid[0, 6])
    b5_plot = fig.add_subplot(grid[1, 2])
    b6_plot = fig.add_subplot(grid[1, 3])
    b7_plot = fig.add_subplot(grid[1, 4])
    b8_plot = fig.add_subplot(grid[1, 5])
    b9_plot = fig.add_subplot(grid[1, 6])
    b10_plot = fig.add_subplot(grid[2, 2])
    b11_plot = fig.add_subplot(grid[2, 3])
    b12_plot = fig.add_subplot(grid[2, 4])
    b13_plot = fig.add_subplot(grid[2, 5])
    b14_plot = fig.add_subplot(grid[2, 6])
    b15_plot = fig.add_subplot(grid[3, 2])
    b16_plot = fig.add_subplot(grid[3, 3])
    b17_plot = fig.add_subplot(grid[3, 4])
    b18_plot = fig.add_subplot(grid[3, 5])
    b19_plot = fig.add_subplot(grid[3, 6])

    x = np.load(data_dir / conf.processed_dir / conf.split / x_fname)[:,:,conf.use_channels]
    x_plot.imshow(x[:,:,[4,2,1]]/255)
    x_plot.axis("off")
    x_plot.set_title("False color composite (B5, B4, B2)")
    #plt.imshow(x[:,:,[4,2,1]]/255)
    #plt.savefig("./sailencymap/sailency_x_true.png")

    mask = np.sum(x[:,:,:7], axis=2) == 0
    if conf.normalize == "mean-std":
        x = (x - _mean) / _std
    if conf.normalize == "min-max":
        x = (x - _min) / (_max - _min)
    
    y_fname = x_fname.replace("tiff", "mask")
    y_true = np.load(data_dir / conf.processed_dir / conf.split / y_fname) + 1
    y_true[mask] = 0
    #plt.imshow(y_true)
    #plt.savefig("./sailencymap/sailency_y_true.png")
    y_plot.imshow(y_true)
    y_plot.axis("off")
    y_plot.set_title("Label")

    x = torch.from_numpy(np.expand_dims(x, axis=0)).float()
    
    x = x.permute(0, 3, 1, 2).to(device)
    x = x.requires_grad_()
    y = model(x)
    y = y.permute(0, 2, 3, 1)
    y = torch.nn.Softmax(3)(y)

    _y = y
    plt.imshow(_y.detach().cpu().numpy()[0,:,:,1])
    plt.savefig("./sailencymap/sailency_y_pred.png")
    y[:,:,:,1].mean().backward()
    _x = x.grad.data.abs().detach().cpu().numpy()[0].transpose(1,2,0)

    for i in range(_x.shape[2]):
        #plt.figure()
        #plt.imshow(_x[:,:,i].clip(0,1), cmap="hot")
        varname = f"b{i}_plot"
        globals()[varname].imshow(_x[:,:,i], cmap="hot")
        globals()[varname].axis("off")
        globals()[varname].set_title(channels[i])
        #plt.savefig(f"./sailencymap/channel_{i}.png")
        print(f"channel={i}, sum={np.sum(_x[:,:,i])}")
    
    #_x = _x.sum(axis=2)
    #plt.figure()
    #plt.imshow(_x, cmap="hot")
    plt.tight_layout()
    #fig.suptitle("Feature-wise saliency for debris glacier segmentation")
    plt.savefig(f"./sailencymap/subplot.png")

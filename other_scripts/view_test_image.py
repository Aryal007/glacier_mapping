import numpy as np
import matplotlib.pyplot as plt
import pathlib
from matplotlib import cm 
from matplotlib.colors import ListedColormap
import warnings 
top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Blues', 128)
newcolors = np.vstack((top(np.linspace(0, 0.7, 128)),
                       bottom(np.linspace(0.3, 1, 128))))
OrangeBlue = ListedColormap(newcolors, name='OrangeBlue')
warnings.filterwarnings("ignore")

filename = "mask_16_slice_31.npy"
model = "7_channels"
threshold = 0.53

imagename = filename.replace("mask", "tiff")
filedir = pathlib.Path("/mnt/datadrive/noaa/test_images/processed/")
preddir = pathlib.Path("/mnt/datadrive/noaa/test_images/preds/"+model)

fig, plots = plt.subplots(nrows = 1, ncols=3)

plots[0].imshow(np.load(filedir / imagename)[:,:,:3])
plots[0].axis('off')
plots[1].imshow(np.load(filedir / filename), cmap = OrangeBlue, vmin=0, vmax=1)
plots[1].axis('off')
plots[2].imshow(np.load(preddir / filename) >= threshold, cmap = OrangeBlue, vmin=0, vmax=1)
plots[2].axis('off')

plt.savefig("image_"+filename.split(".")[0]+".png")
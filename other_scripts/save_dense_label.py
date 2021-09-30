import numpy as np
import matplotlib.pyplot as plt
import warnings
import os, sys, pathlib
sys.path.insert(0,os.path.normpath(os.getcwd() + os.sep + os.pardir))
import coastal_mapping.data.slice as fn
import matplotlib
from matplotlib import cm 
from matplotlib.colors import ListedColormap

top = cm.get_cmap('Oranges', 128)
bottom = cm.get_cmap('Blues', 128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)[::-1]),
                       bottom(np.linspace(0, 1, 128))))
OrangeBlue = ListedColormap(newcolors, name='OrangeBlue')

if __name__ == "__main__":

    filename = "2017_NOAA_ANWR_4Band11"
    sparse = True
    data_dir = data_dir = pathlib.Path("/mnt/datadrive/noaa/")
    
    if sparse:
        tif_path = data_dir / "images" / (filename+".TIF")
        labels_path = data_dir / "labels" / (filename+".shp")
    else:
        tif_path = data_dir / "test_images" / "images" / (filename+".TIF")
        labels_path = data_dir / "test_images" / "labels" / (filename+".shp")

    shp = fn.read_shp(labels_path)
    tiff = fn.read_tiff(tif_path)
    image = tiff.read().transpose(1,2,0)[:,:,:3]
    mask = fn.get_mask(tiff, shp)
    empty = np.sum(mask, axis=2) == 0
    if sparse:
        mask = np.argmax(mask, axis=2)
        mask[mask == 1] = 2 
        mask[empty] = 1
    plt.imshow(mask, cmap=OrangeBlue, alpha=0.7)
    plt.axis('off')
    plt.savefig(filename+'.png')
    print("Saved")
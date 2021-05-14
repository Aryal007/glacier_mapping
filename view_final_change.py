from PIL import Image
import numpy as np
import os, pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["black","silver", "lightgreen", "forestgreen", "cyan", "tan", "dodgerblue", "white"])

inp_dir = pathlib.Path("./final")
out_dir = pathlib.Path("./final_vis")

files = sorted(os.listdir(inp_dir))

for f in files:
    arr = np.array(Image.open(inp_dir / f))
    #plt.imshow(arr, cmap=cmap)
    plt.imsave(out_dir / f, arr, cmap=cmap)
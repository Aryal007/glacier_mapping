import numpy as np
import matplotlib.pyplot as plt
import os, pathlib
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["black", "silver", "lightgreen", "forestgreen", "cyan", "tan", "dodgerblue", "white"])

data_dir = pathlib.Path("/datadrive/DynamicEarthNet/processed/train")

for files in sorted(os.listdir(data_dir)):
	if "tiff" in files:
		tiff = np.load(data_dir / files)
		labels = np.load(data_dir / files.replace("tiff", "mask"))
		fig, ax = plt.subplots(1,2)
		ax[0].imshow(tiff[:,:,:3])
		mask = np.sum(labels, axis=2) == 0
		labels = np.argmax(labels, axis=2)
		labels[mask] = 0
		ax[1].imshow(labels, cmap=cmap)
		plt.tight_layout()
		plt.savefig("./temp/"+files.replace("npy","png"))
		plt.close()



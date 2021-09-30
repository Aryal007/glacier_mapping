import numpy as np 
import pathlib
import matplotlib.pyplot as plt 

method = "ndswi"
data_dir = pathlib.Path("/mnt/datadrive/noaa/hist_arr/")
landfilename = method+"_land.npy"
waterfilename = method+"_water.npy"

land = np.load(data_dir / landfilename)
water = np.load(data_dir / waterfilename)

plt.hist(water, bins=100, alpha=0.5, density = True, label="Water", log=True)
plt.hist(land, bins=100, alpha=0.5, density = True, label="Land", log=True)
plt.axvline(0.52, color='r', label=method+" threshold")
plt.text(0.54, 35, "0.52", color='r')
plt.xlim(-1, 1)
#plt.ylim(100)
plt.title("NDSWI")
plt.legend(loc='upper left')
plt.ylabel('Percentage')
plt.xlabel('Intensity')
plt.savefig(method+".png")
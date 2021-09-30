import numpy as np 
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
import yaml, json, pathlib, os
import pdb

data_dir = pathlib.Path("/mnt/datadrive/noaa/")
save_location = pathlib.Path("/mnt/datadrive/noaa/hist_arr/")
x_y_dir = data_dir / "processed" / "val"

x_filenames = [x for x in os.listdir(x_y_dir) if "tiff" in x] 

for x_filename in x_filenames:
    y_filename = x_filename.replace("tiff", "mask")
    mask = np.sum(np.load(x_y_dir / y_filename), axis=2) == 1
    true = np.load(x_y_dir / y_filename)[:,:,1][mask]
    try:
        ndwi_land = np.hstack((ndwi_land, np.load(x_y_dir / x_filename)[:,:,5][mask][true == 0]))
        ndwi_water = np.hstack((ndwi_water, np.load(x_y_dir / x_filename)[:,:,5][mask][true == 1]))
        #ndswi_land = np.hstack((ndswi_land, np.load(x_y_dir / x_filename)[:,:,6][mask][true == 0]))
        #ndswi_water = np.hstack((ndswi_water, np.load(x_y_dir / x_filename)[:,:,6][mask][true == 1]))
        #rf_land = np.hstack((rf_land, np.load(data_dir / "processed" / "val_preds" / "random_forest" / y_filename)[mask][true == 0]))
        #rf_water = np.hstack((rf_water, np.load(data_dir / "processed" / "val_preds" / "random_forest" / y_filename)[mask][true == 1]))
        xgboost_land = np.hstack((xgboost_land, np.load(data_dir / "processed" / "val_preds" / "xgboost" / y_filename)[mask][true == 0]))
        xgboost_water = np.hstack((xgboost_water, np.load(data_dir / "processed" / "val_preds" / "xgboost" / y_filename)[mask][true == 1]))
        unet_4_land = np.hstack((unet_4_land, np.load(data_dir / "processed" / "val_preds" / "4_channels" / y_filename)[mask][true == 0]))
        unet_4_water = np.hstack((unet_4_water, np.load(data_dir / "processed" / "val_preds" / "4_channels" / y_filename)[mask][true == 1]))
        unet_7_land = np.hstack((unet_7_land, np.load(data_dir / "processed" / "val_preds" / "7_channels" / y_filename)[mask][true == 0]))
        unet_7_water = np.hstack((unet_7_water, np.load(data_dir / "processed" / "val_preds" / "7_channels" / y_filename)[mask][true == 1]))
    except Exception as e:
        ndwi_land = np.load(x_y_dir / x_filename)[:,:,5][mask][true == 0]
        ndwi_water = np.load(x_y_dir / x_filename)[:,:,5][mask][true == 1]
        #ndswi_land = np.load(x_y_dir / x_filename)[:,:,6][mask][true == 0]
        #ndswi_water = np.load(x_y_dir / x_filename)[:,:,6][mask][true == 1]
        #rf_land = np.load(data_dir / "processed" / "val_preds" / "random_forest" / y_filename)[mask][true == 0]
        #rf_water = np.load(data_dir / "processed" / "val_preds" / "random_forest" / y_filename)[mask][true == 1]
        xgboost_land = np.load(data_dir / "processed" / "val_preds" / "xgboost" / y_filename)[mask][true == 0]
        xgboost_water = np.load(data_dir / "processed" / "val_preds" / "xgboost" / y_filename)[mask][true == 1]
        unet_4_land = np.load(data_dir / "processed" / "val_preds" / "4_channels" / y_filename)[mask][true == 0]
        unet_4_water = np.load(data_dir / "processed" / "val_preds" / "4_channels" / y_filename)[mask][true == 1]
        unet_7_land = np.load(data_dir / "processed" / "val_preds" / "7_channels" / y_filename)[mask][true == 0]
        unet_7_water = np.load(data_dir / "processed" / "val_preds" / "7_channels" / y_filename)[mask][true == 1]

np.save( save_location / "ndwi_land", ndwi_land)
np.save( save_location / "ndwi_water", ndwi_water)
np.save( save_location / "ndswi_land", ndswi_land)
np.save( save_location / "ndswi_water", ndswi_water)
np.save( save_location / "rf_land", rf_land)
np.save( save_location / "rf_water", rf_water)
np.save( save_location / "xgboost_land", xgboost_land)
np.save( save_location / "xgboost_water", xgboost_water)
np.save( save_location / "unet_4_land", unet_4_land)
np.save( save_location / "unet_4_water", unet_4_water)
np.save( save_location / "unet_7_land", unet_7_land)
np.save( save_location / "unet_7_water", unet_7_water)
print("Saved")
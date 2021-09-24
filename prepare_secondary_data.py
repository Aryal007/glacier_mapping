import glob, os, pdb
from pathlib import Path
import numpy as np
from tifffile import imread

val_ids = ['hxu', 'jja', 'kuo', 'pxs', 'qus']
data_dir = Path('./data')
out_dir = data_dir / 'processed'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(out_dir / 'train'):
    os.mkdir(out_dir / 'train')
if not os.path.exists(out_dir / 'val'):
    os.mkdir(out_dir / 'val')

if __name__ == "__main__":
    outputs_dir = [x for x in os.listdir(data_dir) if "outputs" in x]
    for output_dir in outputs_dir:
        output_prefix = output_dir.split("_")[-1]
        np_files = data_dir / output_dir / "*.npy"
        np_files = glob.glob(str(np_files))
        for np_file in np_files:
            flag_train = True
            fname = np_file.split("/")[-1].split(".")[0]
            vv_fname = fname+"_vv.tif"
            vh_fname = fname+"_vh.tif"
            vv_fname = data_dir / 'train_features' / vv_fname
            vh_fname = data_dir / 'train_features' / vh_fname
            vv = imread(vv_fname)
            vh = imread(vh_fname)
            pred = np.load(np_file)
            out_arr = np.concatenate((np.expand_dims(vv, axis=2), np.expand_dims(vh, axis=2), np.expand_dims(pred, axis=2)), axis=2)
            mask_fname = fname + ".tif"
            mask_fname = data_dir / 'train_labels' / mask_fname
            mask_arr = imread(mask_fname)
            for val_id in val_ids:
                if val_id in fname:
                    flag_train = False
            tiff_out_fname = "tiff_" + fname + "_" +output_prefix
            mask_out_fname = "mask_" + fname + "_" +output_prefix
            if flag_train:
                np.save(out_dir / 'train' / tiff_out_fname, out_arr)
                np.save(out_dir / 'train' / mask_out_fname, mask_arr)
            else:
                np.save(out_dir / 'val' / tiff_out_fname, out_arr)
                np.save(out_dir / 'val' / mask_out_fname, mask_arr)
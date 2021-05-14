import os, yaml, pathlib, warnings
import numpy as np
from addict import Dict
import coastal_mapping.data.slice as fn
warnings.filterwarnings("ignore")
np.random.seed(7)

conf = Dict(yaml.safe_load(open('./conf/get_final_slice.yaml')))
window_size = conf.window_size
processed_dir = pathlib.Path(conf.processed_dir)
label = conf.label
negative_label = conf.negative_label

labels_dict = {
        1: "Impervious",
        2: "Agriculture",
        3: "Forest",
        4: "Wetlands",
        5: "Soil",
        6: "Water",
        7: "Snow"
    }
copy_dirs = ["train", "val"]
out_dir = processed_dir / labels_dict[label]
fn.remove_and_create(out_dir)
weights = np.zeros(2)
neg_size = conf.neg_size
rand_size = conf.rand_size

n_train, n_val = 0,0
for directory in copy_dirs:
    cur_dir = out_dir / directory
    fn.remove_and_create(cur_dir)
    print(f"Working on {cur_dir}")
    files = os.listdir(processed_dir / directory)
    if conf.augmented:
        gen = [x for x in files if "mask" in x]
    else:
        gen = [x for x in files if x.startswith("mask")]
    print(f"Number of files: {len(gen)}")
    for fname in gen:
        mask_arr = np.load(processed_dir / directory / fname)
        if (np.sum((mask_arr==label).astype(np.bool)) / len(mask_arr.flatten())) < 0.02:      # Assuming negative samples
            if directory == "train":                                                         # negative samples kind of unnecessary for validation?
                if fname.startswith("mask"):
                    if any(neg in mask_arr.flatten() for neg in negative_label):
                        randint = np.random.random_integers(20)
                        if randint < neg_size:
                            out_mask = np.zeros((window_size[0], window_size[1], 2))
                            out_mask[:,:,0][mask_arr == label] = 1
                            out_mask[:,:,1][np.logical_and(mask_arr != label, mask_arr != 0)] = 1
                            np.save(cur_dir / fname, out_mask)
                            tiff_fname = fname.replace("mask", "tiff")
                            tiff = np.load(processed_dir / directory / tiff_fname)
                            np.save(cur_dir / tiff_fname, tiff)
                            if directory == "train":
                                weights += np.sum(out_mask, axis=(0,1))/(window_size[0] * window_size[1])
                                n_train += 1
                            else:
                                n_val += 1
                    else:
                        randint = np.random.random_integers(20)
                        if randint < rand_size:
                            out_mask = np.zeros((window_size[0], window_size[1], 2))
                            out_mask[:,:,0][mask_arr == label] = 1
                            out_mask[:,:,1][np.logical_and(mask_arr != label, mask_arr != 0)] = 1
                            np.save(cur_dir / fname, out_mask)
                            tiff_fname = fname.replace("mask", "tiff")
                            tiff = np.load(processed_dir / directory / tiff_fname)
                            np.save(cur_dir / tiff_fname, tiff)
                            if directory == "train":
                                weights += np.sum(out_mask, axis=(0,1))/(window_size[0] * window_size[1])
                                n_train += 1
                            else:
                                n_val += 1
        else:
            out_mask = np.zeros((window_size[0], window_size[1], 2))
            out_mask[:,:,0][mask_arr == label] = 1
            out_mask[:,:,1][np.logical_and(mask_arr != label, mask_arr != 0)] = 1
            np.save(cur_dir / fname, out_mask)
            tiff_fname = fname.replace("mask", "tiff")
            tiff = np.load(processed_dir / directory / tiff_fname)
            np.save(cur_dir / tiff_fname, tiff)
            if directory == "train":
                weights += np.sum(out_mask, axis=(0,1))/(window_size[0] * window_size[1])
                n_train += 1
            else:
                n_val += 1

weights = np.prod(weights)/weights
weights = weights/np.min(weights)
print(f"Weights for {labels_dict[label]}: {weights}")
print(f"Number of samples:\n\tTraining: {n_train}, Validation: {n_val}")
np.save(out_dir/"weights", weights)
normalize = np.load(processed_dir / "normalize.npy")
np.save(out_dir/"normalize", normalize)
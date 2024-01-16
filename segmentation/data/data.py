#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:24:56 2021

@author: mibook
"""
import glob, os, random, torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import rasterio
import pdb

def fetch_loaders(processed_dir, batch_size=32, use_channels=[0,1], normalize=False, train_folder='train_satellite', shuffle=True):
    """ Function to fetch dataLoaders for the Training / Validation
    Args:
        processed_dir(str): Directory with the processed data
        batch_size(int): The size of each batch during training. Defaults to 32.
    Return:
        Returns train and test dataloaders
    """
    dataset = KelpDataset(processed_dir / train_folder, use_channels, normalize,
                                   transforms=transforms.Compose([
                                       FlipHorizontal(0.15),
                                       FlipVertical(0.15),
                                       Rot270(0.15),
                                    ])
                                    )
    
    n_data = len(dataset)
    val_split, test_split = 0.2, 0.2
    n_test = int(n_data*test_split)
    n_val = int(n_data*val_split)
    n_train = n_data-n_test-n_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class KelpDataset(Dataset):
    """Custom Dataset for Kelp Data
    Indexing the i^th element returns the underlying image and the associated
    binary mask
    """

    def __init__(self, folder_path, use_channels, normalize, transforms=None):
        """Initialize dataset.
        Args:
            folder_path(str): A path to data directory
        """

        self.img_files = glob.glob(os.path.join(folder_path, '*tif*'))[:1000]
        self.mask_files = [s.replace("satellite", "kelp") for s in self.img_files]
        self.use_channels = use_channels
        self.normalize = normalize
        self.transforms = transforms
        try:
            arr = np.load(folder_path.parent / "normalize_train.npy")
        except:
            print("Normalize train does not exist, computing")
            samples = self.img_files[:100]
            means, stds, mins, maxs = [], [], [], []
            arr = []
            for sample in samples:
                x = rasterio.open(sample)
                x_arr = x.read()
                data = np.transpose(x_arr, (1,2,0))
                means.append(np.mean(data, axis=(0,1)))
                stds.append(np.std(data, axis=(0,1)))
                mins.append(np.min(data, axis=(0,1)))
                maxs.append(np.max(data, axis=(0,1)))
            arr.append(np.mean(means, axis=0))
            arr.append(np.mean(stds, axis=0))
            arr.append(np.min(mins, axis=0))
            arr.append(np.max(maxs, axis=0))
            arr = np.asarray(arr)
            print(arr.shape)
            np.save(folder_path.parent / "normalize_train.npy", arr)
        if self.normalize == "min-max":
            self.min, self.max = arr[2][use_channels], arr[3][use_channels]
        if self.normalize == "mean-std":
            self.mean, self.std = arr[0], arr[1]
            self.mean, self.std = self.mean[use_channels], self.std[use_channels]

    def __getitem__(self, index):
        """ getitem method to retrieve a single instance of the dataset
        Args:
            index(int): Index identifier of the data instance
        Return:
            data(x) and corresponding label(y)
        """
        x = rasterio.open(self.img_files[index])
        x_arr = x.read()
        data = np.transpose(x_arr, (1,2,0))
        data = data[:, :, self.use_channels]
        if self.normalize == "min-max":
            data = np.clip(data, self.min, self.max)
            data = (data - self.min) / (self.max - self.min)
        elif self.normalize == "mean-std":
            data = (data - self.mean) / self.std
        else:
            raise ValueError("normalize must be min-max or mean-std")
        y = rasterio.open(self.mask_files[index])
        pos = y.read(1)
        neg = ~pos+2
        label = np.concatenate((np.expand_dims(pos, axis=2),np.expand_dims(neg, axis=2)), axis=2)
        if self.transforms:
            sample = {'image': data, 'mask': label}
            sample = self.transforms(sample)
            data = torch.from_numpy(sample['image'].copy()).float()
            label = torch.from_numpy(sample['mask'].copy()).float()
        else:
            data = torch.from_numpy(data).float()
            label = torch.from_numpy(label).float()
        return data, label

    def __len__(self):
        """ Function to return the length of the dataset
            Args:
                None
            Return:
                len(img_files)(int): The length of the dataset (img_files)
        """
        return len(self.img_files)


class FlipHorizontal(object):
    """Flip horizontal randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of FlipHorizontal
    """

    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data = data[:, ::-1, :]
            label = label[:, ::-1, :]
        return {'image': data, 'mask': label}


class FlipVertical(object):
    """Flip vertically randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of FlipVertical
    """

    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data = data[::-1, :, :]
            label = label[::-1, :, :]
        return {'image': data, 'mask': label}


class Rot270(object):
    """Flip vertically randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of Rot270
    """

    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data = data.transpose((1, 0, 2))
            label = label.transpose((1, 0, 2))
        return {'image': data, 'mask': label}

class DropoutChannels(object):
    """
    Apply Random channel dropouts
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            rand_channel_index = np.random.randint(low=0, high=data.shape[2], size=int(data.shape[2]/5))
            data[:, :, rand_channel_index] = 0
        return {'image': data, 'mask': label}

class ElasticDeform(object):
    """
    Apply Elasticdeform from U-Net
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        label = label.astype(np.float32)
        if torch.rand(1) < self.p:
            [data, label] = elasticdeform.deform_random_grid([data, label], axis=(0, 1))
        label = np.round(label).astype(bool)
        return {'image': data, 'mask': label}
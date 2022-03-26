#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:26:56 2021

@author: mibook
"""
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon, box
from rasterio.features import rasterize
from shapely.ops import cascaded_union
import numpy as np
from pathlib import Path
import os
import shutil
import pdb
from skimage.color import rgb2hsv


def read_shp(filename):
    """
    This function reads the shp file given
    filename and returns the geopandas object
    Parameters
    ----------
    filename : string
    Returns
    -------
    geopandas dataframe

    """
    shapefile = gpd.read_file(filename)

    return shapefile


def read_tiff(filename):
    """
    This function reads the tiff file given
    filename and returns the rasterio object
    Parameters
    ----------
    filename : string
    Returns
    -------
    rasterio tiff object

    """
    dataset = rasterio.open(filename)

    return dataset


def check_crs(crs_a, crs_b, verbose=False):
    """
    Verify that two CRS objects Match
    :param crs_a: The first CRS to compare.
        :type crs_a: rasterio.crs
    :param crs_b: The second CRS to compare.
        :type crs_b: rasterio.crs
    :side-effects: Raises an error if the CRS's don't agree
    """
    if verbose:
        print("CRS 1: " + crs_a.to_string() + ", CRS 2: " + crs_b.to_string())
    if rasterio.crs.CRS.from_string(
            crs_a.to_string()) != rasterio.crs.CRS.from_string(
            crs_b.to_string()):
        raise ValueError("Coordinate reference systems do not agree")


def get_mask(tiff, shp, column="Glaciers"):
    """
    This function reads the tiff filename and associated
    shp filename given and returns the numpy array mask
    of the labels
    Parameters
    ----------
    tiff : rasterio.io.DatasetReader
    shp : geopandas.geodataframe.GeoDataFrame
    Returns
    -------
    numpy array of shape (channels * width * height)

    """

    # Generate polygon
    def poly_from_coord(polygon, transform):
        """
        Get a transformed polygon
        https://lpsmlgeo.github.io/2019-09-22-binary_mask/
        """
        poly_pts = []
        poly = cascaded_union(polygon)
        for i in np.array(poly.exterior.coords):
            # in case polygonz format
            poly_pts.append(~transform * tuple(i)[:2])
        return Polygon(poly_pts)

    # Clip shapefile
    def clip_shapefile(img_bounds, img_meta, shp):
        """
        Clip Shapefile Extents to Image Bounding Box
        :param img_bounds: The rectangular lat/long bounding box associated with a
            raster tiff.
        :param img_meta: The metadata field associated with a geotiff. Expected to
            contain transform (coordinate system), height, and width fields.
        :param shps: A list of K geopandas shapefiles, used to build the mask.
            Assumed to be in the same coordinate system as img_data.
        :return result: The same shapefiles as shps, but with polygons that don't
            overlap the img bounding box removed.
        """
        bbox = box(*img_bounds)
        bbox_poly = gpd.GeoDataFrame({'geometry': bbox}, index=[
                                     0], crs=img_meta["crs"].data)
        return shp.loc[shp.intersects(bbox_poly["geometry"][0])]
    classes = sorted(list(set(shp[column])))
    print(f"Classes = {classes}")

    shapefile_crs = rasterio.crs.CRS.from_string(str(shp.crs))

    if shapefile_crs != tiff.meta["crs"]:
        shp = shp.to_crs(tiff.meta["crs"].data)
    check_crs(tiff.crs, shp.crs)
    shapefile = clip_shapefile(tiff.bounds, tiff.meta, shp)
    mask = np.zeros((tiff.height, tiff.width, len(classes)))

    for key, value in enumerate(classes):
        geom = shapefile[shapefile[column] == value]
        poly_shp = []
        im_size = (tiff.meta['height'], tiff.meta['width'])
        for num, row in geom.iterrows():
            if row['geometry'].geom_type == 'Polygon':
                poly_shp.append(
                    poly_from_coord(
                        row['geometry'],
                        tiff.meta['transform']))
            else:
                for p in row['geometry']:
                    poly_shp.append(poly_from_coord(p, tiff.meta['transform']))
        try:
            channel_mask = rasterize(shapes=poly_shp, out_shape=im_size)
            mask[:, :, key] = channel_mask
        except Exception as e:
            pass
            # Already 0 if no mask

    return mask


def add_index(tiff_np, index1, index2):
    rsi = (tiff_np[:, :, index1] - tiff_np[:, :, index2]) / \
        (tiff_np[:, :, index1] + tiff_np[:, :, index2])
    rsi = np.nan_to_num(rsi).clip(-1, 1)
    tiff_np = np.concatenate((tiff_np, np.expand_dims(rsi, axis=2)), axis=2)
    return tiff_np


def save_slices(
        filename,
        filenum,
        tiff,
        mask,
        roi_mask,
        savepath,
        saved_df,
        **conf):
    _mask = np.zeros((mask.shape[0], mask.shape[1]))
    roi_mask = np.squeeze(roi_mask) == 0
    for i in range(mask.shape[2]):
        _mask[mask[:, :, i] == 1] = i + 1
    mask = _mask.astype(np.uint8)

    def verify_slice_size(slice, conf):
        if slice.shape[0] != conf["window_size"][0] or slice.shape[1] != conf["window_size"][1]:
            if len(slice.shape) == 2:
                temp = np.zeros(
                    (conf["window_size"][0], conf["window_size"][1]))
                temp[0:slice.shape[0], 0:slice.shape[1]] = slice
            else:
                temp = np.zeros(
                    (conf["window_size"][0],
                     conf["window_size"][1],
                        slice.shape[2]))
                temp[0:slice.shape[0], 0:slice.shape[1], :] = slice
            slice = temp
        return slice

    def filter_percentage(slice, percentage, type="mask"):
        if type == "image":
            labelled_pixels = np.sum(np.sum(slice[:, :, :5], axis=2) != 0)
            percentage = 0.5
        else:
            labelled_pixels = np.sum(slice != 0)
        total_pixels = slice.shape[0] * slice.shape[1]
        if labelled_pixels / total_pixels < percentage:
            return False
        return True

    def save_slice(arr, filename):
        np.save(filename, arr)

    def get_pixel_count(tiff_slice, mask_slice):
        mas = np.sum(tiff_slice[:, :, :5], axis=2) == 0
        mask_slice[mas] = 0
        deb, ci = np.sum(mask_slice == 1), np.sum(mask_slice == 2)
        mas = np.sum(mas)
        bg = mask_slice.shape[0] * mask_slice.shape[1] - (ci + deb + mas)
        return bg, ci, deb, mas

    if not os.path.exists(conf["out_dir"]):
        os.makedirs(conf["out_dir"])

    tiff_np = np.transpose(tiff.read(), (1, 2, 0))
    tiff_np = tiff_np[:, :, conf["use_bands"]]
    tiff_np[roi_mask] = 0
    tiff_np = tiff_np.astype(np.float32)

    if conf["add_ndvi"]:
        tiff_np = add_index(tiff_np, index1=3, index2=2)
    if conf["add_ndwi"]:
        tiff_np = add_index(tiff_np, index1=3, index2=4)
    if conf["add_ndsi"]:
        tiff_np = add_index(tiff_np, index1=1, index2=4)
    if conf["add_hsv"]:
        rgb_img = tiff_np[:, :, :3] / 255
        hsv_img = rgb2hsv(rgb_img[:, :, [2, 1, 0]])
        tiff_np = np.concatenate((tiff_np, hsv_img), axis=2)

    slicenum = 0
    for row in range(
            0,
            tiff_np.shape[0],
            conf["window_size"][0] -
            conf["overlap"]):
        for column in range(
                0,
                tiff_np.shape[0],
                conf["window_size"][1] -
                conf["overlap"]):
            mask_slice = mask[row:row + conf["window_size"]
                              [0], column:column + conf["window_size"][1]]
            mask_slice = verify_slice_size(mask_slice, conf)

            if filter_percentage(mask_slice, conf["filter"]):
                tiff_slice = tiff_np[row:row +
                                     conf["window_size"][0], column:column +
                                     conf["window_size"][1], :]
                tiff_slice = verify_slice_size(tiff_slice, conf)
                if filter_percentage(tiff_slice, conf["filter"], type="image"):
                    mask_fname, tiff_fname = "mask_" + str(filenum) + "_slice_" + str(
                        slicenum), "tiff_" + str(filenum) + "_slice_" + str(slicenum)
                    bg, ci, deb, mas = get_pixel_count(tiff_slice, mask_slice)
                    _tot = bg + ci + deb + mas
                    _row = [filename, filenum, slicenum,
                            bg, ci, deb, mas,
                            bg / _tot, ci / _tot, deb / _tot, mas / _tot,
                            os.path.basename(savepath)]
                    saved_df.loc[len(saved_df.index)] = _row
                    save_slice(mask_slice, savepath / mask_fname)
                    save_slice(tiff_slice, savepath / tiff_fname)
                    print(f"Saved image {filenum} slice {slicenum}")
            slicenum += 1

    tiff_np = np.nan_to_num(tiff_np.astype(np.float64))
    if np.isnan(np.sum(np.std(tiff_np, axis=(0, 1)))):
        print("***********NaN Value encountered**************")
    return np.mean(
        tiff_np, axis=(
            0, 1)), np.std(
        tiff_np, axis=(
            0, 1)), np.min(
        tiff_np, axis=(
            0, 1)), np.max(
        tiff_np, axis=(
            0, 1)), saved_df


def remove_and_create(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)

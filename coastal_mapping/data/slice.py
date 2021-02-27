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
import os, shutil

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

def check_crs(crs_a, crs_b, verbose = False):
    """
    Verify that two CRS objects Match
    :param crs_a: The first CRS to compare.
        :type crs_a: rasterio.crs
    :param crs_b: The second CRS to compare.
        :type crs_b: rasterio.crs
    :side-effects: Raises an error if the CRS's don't agree
    """
    if verbose:
        print("CRS 1: "+crs_a.to_string()+", CRS 2: "+crs_b.to_string())
    if rasterio.crs.CRS.from_string(crs_a.to_string()) != rasterio.crs.CRS.from_string(
            crs_b.to_string()):
        raise ValueError("Coordinate reference systems do not agree")

def get_mask(tiff, shp, column="Id"):
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
    
    #Generate polygon
    def poly_from_coord(polygon, transform):
        """
        Get a transformed polygon
        https://lpsmlgeo.github.io/2019-09-22-binary_mask/
        """
        poly_pts = []
        poly = cascaded_union(polygon)
        for i in np.array(poly.exterior.coords):
            poly_pts.append(~transform * tuple(i)[:2]) # in case polygonz format
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
        bbox_poly = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=img_meta["crs"].data)
        return shp.loc[shp.intersects(bbox_poly["geometry"][0])]
    
    classes = set(shp[column])

    shapefile_crs = rasterio.crs.CRS.from_string(str(shp.crs))
    print("Here")
    input()
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
                poly_shp.append(poly_from_coord(row['geometry'], tiff.meta['transform']))
            else:
                for p in row['geometry']:
                    poly_shp.append(poly_from_coord(p, tiff.meta['transform']))
        try:
            channel_mask = rasterize(shapes=poly_shp, out_shape=im_size)
            mask[:,:,key] = channel_mask
        except Exception as e:
            print(e)
            print(value)

    return mask

def save_slices(filenum, tiff, mask, **conf):
    def verify_slice_size(slice, conf):
        if slice.shape[0] != conf["window_size"][0] or slice.shape[1] != conf["window_size"][1]:
            temp = np.zeros((conf["window_size"][0], conf["window_size"][1], slice.shape[2]))
            temp[0:slice.shape[0], 0:slice.shape[1]] = slice
            slice = temp
        return slice

    def filter_percentage(slice, percentage):
        labelled_pixels = np.sum(slice)
        total_pixels = slice.shape[0] * slice.shape[1]
        if labelled_pixels/total_pixels < percentage:
            return False
        return True

    def save_slice(arr, filename):
        np.save(filename, arr)

    if not os.path.exists(conf["out_dir"]):
        os.makedirs(conf["out_dir"])
    tiff_np = np.transpose(tiff.read(), (1,2,0))
    print(tiff_np.shape)
    input()

    slicenum = 0
    for row in range(0, tiff_np.shape[0], conf["window_size"][0]-conf["overlap"]):
        for column in range(0, tiff_np.shape[0], conf["window_size"][1]-conf["overlap"]):
            mask_slice = mask[row:row+conf["window_size"][0], column:column+conf["window_size"][1], :]
            mask_slice = verify_slice_size(mask_slice, conf)

            if filter_percentage(mask_slice, conf["filter"]):
                tiff_slice = tiff_np[row:row+conf["window_size"][0], column:column+conf["window_size"][1], :]
                tiff_slice = verify_slice_size(tiff_slice, conf)
                save_slice(mask_slice, conf["out_dir"]+"mask_"+str(filenum)+"_slice_"+str(slicenum))
                save_slice(tiff_slice, conf["out_dir"]+"tiff_"+str(filenum)+"_slice_"+str(slicenum))
                print(f"Saved image {filenum} slice {slicenum}")

            slicenum += 1

def remove_and_create(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)

def train_test_shuffle(out_dir, train_split, val_split, test_split):
    # Remove existing directory, create new ones
    train_path = out_dir + "train/"
    remove_and_create(train_path)
    val_path = out_dir + "val/"
    remove_and_create(val_path)
    test_path = out_dir + "test/"
    remove_and_create(test_path)

    slices = [x for x in os.listdir(out_dir) if (x.endswith('.npy') and "tiff" in x )]
    n_tiffs = len(slices)
    random_index = np.random.permutation(n_tiffs)
    savepath = train_path
    for count, index in enumerate(random_index):
        if count > int(n_tiffs*train_split):
            savepath = val_path
        if count > int(n_tiffs*(train_split+val_split)):
            savepath = test_path
        tiff_filename = slices[index]
        mask_filename = tiff_filename.replace("tiff","mask")
        shutil.move(out_dir+tiff_filename, savepath+tiff_filename)
        shutil.move(out_dir+mask_filename, savepath+mask_filename)
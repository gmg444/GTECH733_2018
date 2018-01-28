import gdal
import scipy
import scipy.ndimage
from scipy.interpolate import griddata
import numpy as np
import ogr
import os
import pandas as ps
import geopandas as gpd
import glob as gl
import utils as ut

def outfileName(infile,outname):
    dirname = os.path.dirname(infile)
    basename = os.path.basename(infile)
    final = dirname+'/'+ basename[:-4]+outname
    return final

def makeTreeTif(infile):
#Input a lidar data tif file and output a binary raster layer with tress as 1 and all else as 0
    outfile = outfileName(infile,"_treetif.tif")
    ds = gdal.Open(infile)
    #Assign variables to information needed tp create output
    band = ds.GetRasterBand(1)
    myarray = band.ReadAsArray()
    rows = myarray.shape[0]
    cols = myarray.shape[1]
    geotrans = ds.GetGeoTransform()
    proj = ds.GetProjection()
    
    #Create an array of the input raster
    #myarray = np.array(band.ReadAsArray())
    #Create a binary array where buildings(>=50) = 1
    buildingsBinary = np.where(myarray>=50, 1, 0).astype(np.bool)
    # Create a binary array were ground(<=) = 1
    groundBinary = np.where(myarray <= 2, 1, 0).astype(np.bool)

    myarray = None
    #Dilate buildings binary array
    buildingsBinary_dilation = scipy.ndimage.morphology.binary_dilation(buildingsBinary)

    #Add ground binary and building binary to create non tree binary array 
    nonTreeBinary = np.add(buildingsBinary_dilation,groundBinary)
    #Dilate non tree binary array
    nonTreeBinary_dilation = scipy.ndimage.morphology.binary_dilation(nonTreeBinary)
    #Close gaps in dilated non tree binary array to remove isolated 1 values
    nonTreeBinary_dilation_opening =scipy.ndimage.binary_opening(nonTreeBinary_dilation, structure=np.ones((6,6))).astype(np.int)
    #Create binary array where trees = 1
    treeBinary = np.where(nonTreeBinary_dilation_opening==1, 0, 1)
    #Dilate  binary tree array
    treeBinary_dilation = scipy.ndimage.morphology.binary_dilation(treeBinary)
    #Close gaps in dilated tree binary array to remove isolated 1 values
    treeBinary_dilation_opening = scipy.ndimage.binary_opening(treeBinary_dilation, structure=np.ones((5,5))).astype(np.int)
    
    #Assign variables for output file info, name, filetype
    dstfile = outfile
    driver = gdal.GetDriverByName('GTiff')
    #Create new tiff file, write array to tiff file
    dataset = driver.Create(dstfile,cols,rows,1,gdal.GDT_Byte)
    trees = treeBinary_dilation_opening.astype(np.ubyte)
    dataset.GetRasterBand(1).WriteArray(trees)
    #Set spatial reference and projection of output file to same as input file
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)
    dataset.FlushCache()
    dataset = None
    return outfile, trees

def make_tree_shp(in_tiff, out_shp):
    in_file, trees = makeTreeTif(in_tiff)
    in_shp = in_file.replace(".tif", ".shp")
    in_shp = ut.make_polygon(in_file, in_shp, "trees")
    ut.dissolve_polygon(in_shp, out_shp)
    return trees

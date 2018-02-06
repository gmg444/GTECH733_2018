
# -------------------------------------------------------------------------
# Tree classification with GDAL and scikit image. Written by Anne Schwenker
# and Shantal Taveras for GTECH734 Geo Web Services at Hunter College 2017.
# -------------------------------------------------------------------------

import gdal
import scipy
import scipy.ndimage
import numpy as np
import ogr
import geopandas as gpd

def makeTreeTif(infile):

    # Input a lidar data tif file and output a binary raster layer with tress as 1 and all else as 0
    outfile = infile.replace(".tif", "_tree.tif")
    ds = gdal.Open(infile)

    # Assign variables to information needed tp create output
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    rows = arr.shape[0]
    cols = arr.shape[1]
    geotrans = ds.GetGeoTransform()
    proj = ds.GetProjection()
    
    # Create an array of the input raster
    # Create a binary array where buildings(>=50) = 1
    buildingsBinary = np.where(arr >= 30, 1, 0).astype(np.bool)

    # Create a binary array were ground(<=) = 1
    groundBinary = np.where(arr <= 2, 1, 0).astype(np.bool)

    # Dilate buildings binary array
    buildingsBinary_dilation = scipy.ndimage.morphology.binary_dilation(buildingsBinary)

    # Add ground binary and building binary to create non tree binary array
    nonTreeBinary = np.add(buildingsBinary_dilation,groundBinary)

    # Dilate non tree binary array
    nonTreeBinary_dilation = scipy.ndimage.morphology.binary_dilation(groundBinary)

    # Close gaps in dilated non tree binary array to remove isolated 1 values
    nonTreeBinary_dilation_opening =scipy.ndimage.binary_opening(nonTreeBinary_dilation, structure=np.ones((6,6))).astype(np.int)

    # Create binary array where trees = 1
    treeBinary = np.where(nonTreeBinary_dilation_opening==1, 0, 1)

    # Dilate  binary tree array
    treeBinary_dilation = scipy.ndimage.morphology.binary_dilation(treeBinary)

    # Close gaps in dilated tree binary array to remove isolated 1 values
    treeBinary_dilation_opening = scipy.ndimage.binary_opening(treeBinary_dilation, structure=np.ones((5,5))).astype(np.int)
    
    # Assign variables for output file info, name, filetype
    dstfile = outfile
    driver = gdal.GetDriverByName('GTiff')

    # Create new tiff file, write array to tiff file
    dataset = driver.Create(dstfile, cols, rows, 1, gdal.GDT_Byte)
    trees = treeBinary_dilation_opening.astype(np.ubyte)
    dataset.GetRasterBand(1).WriteArray(trees)

    # Set spatial reference and projection of output file to same as input file
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)
    dataset.FlushCache()
    dataset = None
    return outfile, trees


def make_polygon(infile, outfile, name):

    # Input tif file
    ds = gdal.Open(infile)
    tree_band = ds.GetRasterBand(1)
    polylayer = name

    # Output shape file
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(outfile)
    dst_layer = dst_ds.CreateLayer(polylayer, srs=None)

    # Create New Field in output shapefile to assign value to
    newField = ogr.FieldDefn('Value', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    gdal.Polygonize(tree_band, None, dst_layer, 0, [], callback=None)
    return outfile

def dissolve_polygon(infile, outfile, remove_smaller_than=0):

    # Using geopandas polygon dissolve function to create final output
    layer = gpd.read_file(infile)
    layer = layer[layer.Value == 1]
    layer = layer[['geometry', 'Value']]

    # Need to get rid of tiny polygons
    if remove_smaller_than > 0:
        layer = layer[layer.geometry.area > remove_smaller_than]

    dissolved = layer.dissolve(by='Value')
    dissolved.to_file(outfile)
    return outfile

def make_tree_shp(in_tiff, out_shp):
    in_file, trees = makeTreeTif(in_tiff)
    in_shp = in_file.replace(".tif", ".shp")
    in_shp = make_polygon(in_file, in_shp, "trees")
    dissolve_polygon(in_shp, out_shp)
    return trees


if __name__ == "__main__":
    make_tree_shp("sample_height.tif", "sample_tree.shp")
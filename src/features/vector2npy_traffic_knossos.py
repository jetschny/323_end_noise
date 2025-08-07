# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:49:51 2022

@author: RekanS
"""
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

# import pandas as pd
import sys
import geopandas as gpd
from geocube.api.core import make_geocube
# from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
import rasterio
# from rasterio.plot import show
# from rasterio.mask import mask
import os
# import copy
import xarray as xr
from shapely.geometry import box
from numpy import newaxis
# from skimage.transform import resize

# from rasterio.features import rasterize
# from rasterio.transform import from_bounds

plt.close('all')
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1", "TRUE", "True")

if len(sys.argv) >2:
    print("Total number of arguments passed:", len(sys.argv))
    print("\nArguments passed:", end = " ")
    for i in range(0,len(sys.argv) ):
        print(sys.argv[i],"\t", end = " ")
    plot_switch=str2bool(sys.argv[3])
    write_switch=str2bool(sys.argv[4])
else:
    plot_switch=True
    write_switch=True

# plot_switch=False
# write_switch=False
clip_switch=False
interp_switch=True

#"Vienna" #"Pilsen" #"Clermont_Ferrand" #"Riga" "Bordeaux" "Grenoble" "Innsbruck" "Salzburg" "Kaunas" "Limassol"
#"VIE" #"PIL" #"CLF" #"RIG" "BOR" "GRE" "INN" "SAL" "KAU" "LIM" 
city_string_in="Oslo"
city_string_out="OSL" 

# city_string_in=sys.argv[1]
# city_string_out=sys.argv[2]

print("\n######################## \n")
print("Surface absorption feature creation \n")
print("#### Loading file data from city ",city_string_in," (",city_string_out,")")
print("#### Plotting of figures is ",plot_switch," and writing of output files is ",write_switch)

# base_in_folder="/home/sjet/data/323_end_noise/"
# base_out_folder="/home/sjet/data/323_end_noise/"
base_in_folder          ="Z:/NoiseML/2024/city_data_raw/"
base_out_folder         ="Z:/NoiseML/2024/city_data_raw/"
base_out_folder_pic     ="Z:/NoiseML/2024/city_data_pics/"

in_file                 = '_roadlinks_NO011.dbf'
in_file_target          ='_MRoadsLden.tif'
out_file                = "_feat_traffic"


gdf         = gpd.read_file(base_in_folder+city_string_in +"/" + city_string_in+in_file)
img_target  = rasterio.open(base_in_folder+city_string_in +"/" + city_string_in+in_file_target, 'r') 
  
  
print("#### Loading file done\n")

# gdf = gpd.read_file(base_in_folder+filename, mask=polygon)


print("#### Cropping file")
# print('__builtins__' in globals())
# print(globals().get('__builtins__'))

if '__builtins__' not in globals():
    import builtins
    globals()['__builtins__'] = builtins
    

def read_and_crop_shapefile_with_bbox(shapefile_path, bbox, bbox_crs):
    """
    shapefile_path: path to .shp file
    bbox: (minx, miny, maxx, maxy)
    bbox_crs: EPSG code or proj string of the bbox (e.g., 'EPSG:4326')
    """
    # Load shapefile
    gdf = gpd.read_file(shapefile_path)

    # Create GeoDataFrame for bounding box in its own CRS
    bbox_geom = box(*bbox)
    bbox_gdf = gpd.GeoDataFrame(index=[0], geometry=[bbox_geom], crs=bbox_crs)

    # Reproject bounding box to match shapefile CRS
    bbox_gdf = bbox_gdf.to_crs(gdf.crs)

    # Crop
    cropped = gdf[gdf.geometry.intersects(bbox_gdf.geometry.iloc[0])]

    return cropped

if clip_switch:
    # Create a custom polygon from bbox extend of target data
    img_target_bounds=img_target.bounds
   
    img_target_crs=img_target.crs
    # grid1=img_target.read(1)
    with rasterio.open(base_in_folder+city_string_in +"/" + city_string_in+in_file_target) as img_target:
        grid1 = img_target.read(1)
    # with rasterio.open(outfile) as src:
        # read_data = src.read(1)
    # img_target_epsg = img_target_crs.to_epsg()
    img_target_epsg="EPSG:3035"
    # corner_point1=np.array((img_target_bounds[0] , img_target_bounds[1] ))
    # corner_point2=np.array((img_target_bounds[2] , img_target_bounds[3] ))
     
    # polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
                       # (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])
       
    # poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=img.crs)
    bbox_geom = box(*img_target_bounds)
    bbox_gdf = gpd.GeoDataFrame(index=[0], geometry=[bbox_geom], crs=img_target_crs)
    # bbox_gdf = bbox_gdf.to_crs(src.crs)
       
       
    # grid1, out_transform = mask(img, shapes=[polygon], crop=True)
    gdf_clipped = read_and_crop_shapefile_with_bbox(
        base_in_folder+city_string_in +"/" + city_string_in+in_file, 
        img_target_bounds,
        img_target_epsg)
else:
    # grid1=img.read()
    gdf = gpd.read_file(base_in_folder+city_string_in +"/" + city_string_in+in_file)
    # img_clipped=np.array(img)

# grid1=np.squeeze(grid1)

# if interp_switch:
#     # grid1 = grid1[img_target.shape]s
    
#     if city_string_out=="PIL":
#         grid1_0 = np.zeros(img_target.shape)-999.25
#         grid1_0[0:grid1.shape[0],0:(grid1.shape[1]-1)]=grid1[0:grid1.shape[0],0:(grid1.shape[1]-1)]
#         grid1=grid1_0
#     if city_string_out=="SAL":
#         grid1 = grid1[:,1:1+img_target.shape[1]]
#     else :
#         # grid1 = grid1[1:1+img_target.shape[0],1:1+img_target.shape[1]]
#     # grid1 = resize(grid1, img_target.shape)
#         45
print("#### Cropping file done \n")

if plot_switch:
    print("#### Plotting file")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    gdf.plot( ax=ax1, legend=True)
    # poly_gdf.boundary.plot(ax=ax1, color="red")
    # bbox_gdf.boundary.plot(ax=ax1, color='red', linewidth=2)
    # ax1.set_title("All Unclipped World Data", fontsize=20)
    # ax2.set_title("All Unclipped Capital Data", fontsize=20)
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    gdf.plot( ax=ax2, legend=True)
    plt.show()
    plt.savefig(base_out_folder_pic+city_string_out+out_file+"_clip.png")
    print("#### Plotting file done \n")



print("#### Gridding vector file")

# gdf_clipped = gdf_clipped.to_crs(epsg=3035)  # Or your appropriate UTM zone
gdf = gdf.to_crs(epsg=3035)  # Or your appropriate UTM zone

with rasterio.open(base_in_folder + city_string_in + "/" + city_string_in + in_file_target) as src:
    profile = src.profile
    ref_transform = src.transform
    ref_crs = src.crs
    ref_shape = (src.height, src.width)
    

# Create dummy DataArray to serve as template
reference_da = xr.DataArray(
    np.zeros(ref_shape, dtype=np.float32),
    dims=("y", "x"),
    coords={
        "y": np.arange(ref_shape[0]) * ref_transform.e + ref_transform.f,
        "x": np.arange(ref_shape[1]) * ref_transform.a + ref_transform.c
    },
    attrs={"transform": ref_transform, "crs": ref_crs}
)

# Use `like=` to match rasterization to the reference raster
out_grid = make_geocube(
    vector_data=gdf,
    measurements=["trafficvol"],
    like=reference_da
) 
    
# out_grid = make_geocube(
#     vector_data=gdf_clipped,
#     resolution=(-10, 10),
#     measurements=["trafficvol"]
# )

# out_grid = make_geocube(
#     vector_data=gdf_clipped,
#     measurements=["trafficvol"],
#     resolution=(-10, 10),  # optional if you define raster_shape and transform
#     raster_bounds=img_target.bounds,
#     raster_crs=img_target.crs,
#     raster_transform=img_target.transform
# )

print("#### Gridding vector file done \n")

if plot_switch:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    gdf.plot( ax=ax1, legend=True)
    out_grid.trafficvol.plot(ax=ax2)
    
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    plt.show()
    plt.savefig(base_out_folder_pic+city_string_out+out_file+"_grid.png")
    
def in_spyder():
    return any('spyder' in name for name in sys.modules)

if write_switch:
    print("#### Saving to npy file")
    # if clip_switch:
    #     out_file=out_file+"_clip.npy"
    # else:
    #     out_file=out_file+".npy"
    arr=out_grid.trafficvol.values
    arr2=np.nan_to_num(arr, nan=-999.25, posinf=-999.25)
    
    output_npy_path = os.path.join(base_out_folder, city_string_in, city_string_out + out_file + ".npy")
    # np.save(output_npy_path,np.array(out_grid.trafficvol, dtype=np.uint16))
    np.save(output_npy_path,np.array(arr2))
    
    print("#### Saving to npy file ",output_npy_path," done")
   
    out_meta = img_target.meta.copy()
    # epsg_code = int(img.crs.data['init'][5:])
    out_meta.update({"driver": "GTiff",
                     "dtype" : 'float32',
                     "nodata" : -999.25,
                     "crs": img_target.crs,
                     "count": 1})
   
    output_tif_path = os.path.join(base_out_folder, city_string_in, city_string_out + out_file + ".tif")
    
    # index0 = np.where(arr <= 0)
    # grid3[index0]=-999.25
    
    with rasterio.open(output_tif_path, "w", **out_meta) as dest:
        dest.write(arr2[newaxis,:, :])
    
  
  
    print("#### Saving to tiff file ",output_tif_path," done")



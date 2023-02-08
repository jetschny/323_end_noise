#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:18:23 2022

@author: sjet
"""
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import rasterio
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio.mask import mask
import pycrs
from numpy import zeros, newaxis
import matplotlib.colors as colors

plt.close('all')
plot_switch=True
write_switch=True


default_font_size=11
plt.rc('font', size=default_font_size) #controls default text size
plt.rc('axes', titlesize=default_font_size) #fontsize of the title
plt.rc('axes', labelsize=default_font_size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=default_font_size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=default_font_size) #fontsize of the y tick labels
plt.rc('legend', fontsize=default_font_size) #fontsize of the legend

base_in_folder="BCN_data/"

in_grid_file1="2017_isofones_total_dia_mapa_estrategic_soroll_bcn_clip_predRFC02_test.npy"
in_grid_target="2017_isofones_total_dia_mapa_estrategic_soroll_bcn_clip.npy"

out_grid_file="out_noise/2017_isofones_total_dia_mapa_estrategic_soroll_bcn_clip.tif"


img = rasterio.open(base_in_folder+out_grid_file, 'r') 
# Create a custom polygon
corner_point1=np.array((3.656 , 2.06 ))*1e6
corner_point2=np.array((3.669 , 2.074 ))*1e6
# polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
                   (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])

# polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=img.crs)
out_img, out_transform = mask(img, shapes=[polygon], crop=True)

out_meta = img.meta.copy()
# epsg_code = int(img.crs.data['init'][5:])
out_meta.update({"driver": "GTiff",
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "crs": img.crs})


grid1=np.load(base_in_folder+in_grid_file1)

grid_target=np.load(base_in_folder+in_grid_target)
grid_target= grid_target.astype(float)


noise_classes_old=sorted(np.unique(grid_target))
noise_classes_new=[0, 37, 42, 47,52,57,62,67,72,77,80]
# noise_classes_new=np.array(range(0, len(noise_classes_old), 1))

counter=0
for a in noise_classes_old:
    indexxy = np.where(grid_target ==a)
    grid_target[indexxy]=noise_classes_new[counter]
    counter=counter+1

noise_classes_old=sorted(np.unique(grid1))
counter=0
for a in noise_classes_old:
    indexxy = np.where(grid1 ==a)
    grid1[indexxy]=noise_classes_new[counter]
    counter=counter+1
    

# Get the colormap and set the under and bad colors
colMap = plt.cm.get_cmap("gist_rainbow").copy()
colMap.set_under(color='white')

if plot_switch:
    
    cmap = colors.ListedColormap(['white', '#238443','#78C679', '#C2E699', '#FFFFB2', '#FECC5C',
                                  '#FD8D3C','#FF0909', '#B30622', '#67033B', '#1C0054'])
    boundaries = noise_classes_new
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    cmap_diff=colors.ListedColormap(['darkblue','blue', 'white','white', 'red','darkred'])
    boundaries_diff = [-10,-5,-1,0, 1,5,10]             
    norm_diff = colors.BoundaryNorm(boundaries_diff, cmap_diff.N, clip=True)  
    
    cmap_diff2=colors.ListedColormap(['white','green','red','darkred'])
    boundaries_diff2 = [0, 0.1,5,10,15]             
    norm_diff2 = colors.BoundaryNorm(boundaries_diff2, cmap_diff2.N, clip=True)  
        
    fig, axs = plt.subplots(1, 3, figsize=(12, 8))
    
   
    im1=axs[0].imshow(grid_target, cmap=cmap, norm=norm)
    im2=axs[1].imshow(grid1,  cmap=cmap, norm=norm)
    # im3=axs[2].imshow(grid_target-grid1, cmap="seismic")
    im3=axs[2].imshow((grid_target-grid1),cmap=cmap_diff,norm=norm_diff)
    
      # plt.axis('off')
    # plt.contourf(grid1)
    plt.colorbar(im1, ax=axs[0])
    plt.colorbar(im2, ax=axs[1])
    plt.colorbar(im3, ax=axs[2])
    
    axs[0].set_aspect('equal', 'box')
    axs[1].set_aspect('equal', 'box')
    axs[2].set_aspect('equal', 'box')
 
    # im1.set_clim(0.1,80)
    # im2.set_clim(0.1,80)
    # im3.set_clim(-2,2)
    im3.set_clim(0.1,10)
    
    x_window=[600, 800]
    y_window=[1200, 1400]
    # axs[0].set_xlim(x_window)
    # axs[1].set_xlim(x_window)
    # axs[2].set_xlim(x_window)
    
    # axs[0].set_ylim(y_window)
    # axs[1].set_ylim(y_window)
    # axs[2].set_ylim(y_window)
    
    # axs[0].set_title('Distance to Road')
    # axs[1].set_title('Divergence from Topo')
 
    
    # # plt.colorbar(con2, ax=ax2)
    # plt.show()
    
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    
   
    im1=axs.imshow(np.abs(grid_target-grid1),cmap=cmap_diff2, norm=norm_diff2)
    plt.colorbar(im1, ax=axs)

    
    axs.set_aspect('equal', 'box')


    im1.set_clim(0,10)
    
   
    plt.show()

grid_diff=np.abs(grid_target-grid1)
    
with rasterio.open(base_in_folder+out_grid_file, "w", **out_meta) as dest:
    dest.write(grid_target[newaxis,:,:])

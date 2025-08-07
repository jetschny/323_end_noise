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

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import rasterio
# from numpy import newaxis

plt.close('all')
if '__builtins__' not in globals():
    import builtins
    globals()['__builtins__'] = builtins
    
    
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
clip_switch=True
interp_switch=True

#"Vienna" #"Pilsen" #"Clermont_Ferrand" #"Riga" "Bordeaux" "Grenoble" "Innsbruck" "Salzburg" "Kaunas" "Limassol"
#"VIE" #"PIL" #"CLF" #"RIG" "BOR" "GRE" "INN" "SAL" "KAU" "LIM" 
city_string_in="Oslo"
city_string_out="OSL" 

# city_string_in=sys.argv[1]
# city_string_out=sys.argv[2]

print("\n######################## \n")
print("Traffic  feature creation \n")
print("#### Loading file data from city ",city_string_in," (",city_string_out,")")
print("#### Plotting of figures is ",plot_switch," and writing of output files is ",write_switch)

# base_in_folder="/home/sjet/data/323_end_noise/"
# base_out_folder="/home/sjet/data/323_end_noise/"
base_in_folder      ="Z:/NoiseML/2024/city_data_raw/"
base_out_folder     ="Z:/NoiseML/2024/city_data_features/"
base_out_folder_pic ="Z:/NoiseML/2024/city_data_pics/"

in_grid_file1="_feat_traffic.npy"
# in_grid_file2="_raw_osm_roads_nlanes.npy"
in_file_target='_MRoadsLden.tif'
out_file="_feat_traffic_smooth"

grid1=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file1)
grid1=grid1.astype(np.float32)
# grid2=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file1)
img_target = rasterio.open(base_in_folder+city_string_in +"/" + city_string_in+in_file_target, 'r') 

grid_target=img_target.read()
grid_target=np.squeeze(grid_target)
grid_target=grid_target.astype(np.float32)
index0 = np.where(grid_target == img_target.nodata)
grid_target[index0]=0


print("#### Loading npy file done")


print("#### Processing file")

# making all data with "no data value" 0
# index0 = np.where(grid1 == np.min(grid1))
# grid1[index0]=0
# index0 = np.where(grid2 == np.min(grid2))
# grid2[index0]=0


# side_length=10
# radius=divmod(side_length,2)[0]

# grid1_pad=np.pad(grid1, [radius,radius], "symmetric")
# dim_grid1_pad=grid1_pad.shape
# grid1_distance=np.zeros(dim_grid1_pad)
# distance_matrix=np.zeros([radius,radius])

# for indexxy, item in np.ndenumerate(distance_matrix):
#     distance_matrix[indexxy]=1/np.exp(np.sqrt((indexxy[0]-radius)**2 + (indexxy[1]-radius)**2))
    

# def calc_distance(indexxy):
    # return np.mean(grid1_pad[indexxy[0]-radius:indexxy[0]+radius,indexxy[1]-radius:indexxy[1]+radius])  
  
# def check_frame(indexxy):
    # if (min(indexxy)>radius) and (indexxy[0]<(dim_grid1_pad[0]-radius)) and (indexxy[1]<(dim_grid1_pad[1]-radius)):
    #     return calc_distance(indexxy)
    # else:
    #     return 0
       
# for indexxy, item in np.ndenumerate(grid1_pad):
    # grid1_distance[indexxy]=check_frame(indexxy)
    

grid1_smooth = gaussian_filter(grid1, sigma=2)
# grid2_smooth = gaussian_filter(grid2, sigma=2)

# grid1_distance=grid1_distance[radius:-radius, radius:-radius]
# grid1_distance=grid1_distance*(np.median(grid1[np.where(grid1 >0)])/np.median(grid1_distance[np.where(grid1_distance >0)]))
grid3=grid1_smooth

# cropping to initial range of input data
# re-inserting no data value
# grid3[index0]=-999.25
index0 = np.where(grid3 <= 0)
grid3[index0]=-999.25
grid3=grid3.astype(np.float32)

print("#### Processing file done")

print("#### Potting file")

# x = np.linspace(1, grid1.shape[1], grid1.shape[1])
# y = np.linspace(1, grid1.shape[0], grid1.shape[0])
# X, Y = np.meshgrid(x, y)

if write_switch:
    print("#### Saving to output files")
    np.save(base_out_folder+city_string_in+"/"+city_string_out+out_file+".npy",grid3)

    print("... Saving to npy file done")
    
    out_meta = img_target.meta.copy()
    # epsg_code = int(img.crs.data['init'][5:])
    out_meta.update({"driver": "GTiff",
                     "dtype" : 'float32',
                     "nodata" : -999.25,
                     "crs": img_target.crs})
    with rasterio.open(base_out_folder+city_string_in+"/"+city_string_out+out_file+".tif", "w", **out_meta) as dest:
        dest.write(grid3[np.newaxis,:,:])
        
    print("... Saving to tiff file done")
    
if plot_switch:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
    
    im1=ax1.imshow(grid1)
    im2=ax2.imshow(grid1_smooth)
    im3=ax3.imshow(grid_target)
    im1.set_clim(0, np.max(grid1)*1/3)
    im2.set_clim(0, np.max(grid1)*1/3)
    im3.set_clim(40, np.max(grid_target)*3/3)
    
    y_line_slice=1500
    x_dim=np.size(grid_target,axis=0)
    y_dim=np.size(grid_target,axis=1)
    # y_line_slice=1500
    y_line_slice=int(y_dim/2)
    
    plot1=ax1.plot([0,x_dim],[y_line_slice, y_line_slice],'-r')
    plot2=ax2.plot([0,x_dim],[y_line_slice, y_line_slice],'-r')
    plot3=ax3.plot([0,x_dim],[y_line_slice, y_line_slice],'-r')
    
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.colorbar(im3, ax=ax3)
        
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    ax3.set_aspect('equal', 'box')
    
    # ax1.set_xlim(600,800)
    # ax2.set_xlim(600,800)
    
    # ax1.set_ylim(950,1150)
    # ax2.set_ylim(950,1150)
    
    # im2.set_clim(0,2)
    
    # plt.colorbar(con2, ax=ax2)
    plt.savefig(base_out_folder_pic+"/" + city_string_out+out_file+"_streetinfo.png")
    plt.show()
    
    # fig, (axs1)  = plt.subplots(1, 1, figsize=(20, 8))
      
    # axs2 = axs1.twinx()
    # # plot4=axs1.plot(grid4_plot[y_line_slice,:],"-m")
    # plot1=axs1.plot(grid1[y_line_slice,:],"-k")
    # plot2=axs1.plot(grid1_smooth[y_line_slice,:],"-g")
    
    # plot3=axs2.plot(grid2[y_line_slice,:],"--b")
    # plot4=axs2.plot(grid2_smooth[y_line_slice,:],"-c")
    # # plot4=axs1.plot(grid3[y_line_slice,:],"-c")
    
    # axs1.legend(['OSM Speet Limit', 'OSM Speet Limit smoothed'], loc="upper left")
    # axs2.legend(['OSM No Lanes', 'OSM No Lanes smoothed'], loc="upper right")
    
    # # plot5=axs2.plot(grid_target[y_line_slice,:],"-r")
    # # plot7=axs1.plot(grid7_plot[y_line_slice,:],"-y")
    
    # # plt.legend(['OSM Speet Limit', 'OSM Speet Limit proc'], loc="upper left")
    
    # axs1.set_ylabel("OSM Max Speed",fontsize=14)
    # axs1.set_ylim([0, 80])
    # axs2.set_ylabel("OSM Number of langes",fontsize=14)
    # axs2.set_ylim([0, 5])
    
    # plt.savefig(base_out_folder_pic+"/" + city_string_out+out_grid_file+"_maxspeed_nolanes.png")
    # plt.show()
    
    # fig, (axs1)  = plt.subplots(1, 1, figsize=(20, 8))
      
    # axs2 = axs1.twinx()
    
    # plot1=axs1.plot(grid3[y_line_slice,:],"-g")
    # plot2=axs2.plot(grid_target[y_line_slice,:],"-k")
    
    # axs1.legend(['OSM Street Info smoothed'], loc="upper left")
    # axs2.legend(['Target Noise'], loc="upper right")
    
    # axs1.set_ylabel("OSM Street Info smoothed",fontsize=14)
    # axs1.set_ylim([0, 1])
    # axs2.set_ylabel("Target Noise",fontsize=14)
    # axs2.set_ylim([50, 80])
    
    # plt.savefig(base_out_folder_pic+"/" + city_string_out+out_grid_file+"_streetinfo_profile.png")
    # plt.show()
    
    print("#### Potting file done")



    

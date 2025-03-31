# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:09:46 2025

@author: jetschny
"""

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
# import pandas as pd
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import mean_squared_error
# import cv2
# import tensorflow as tf
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from sklearn.metrics.pairwise import cosine_similarity

plt.close('all')
plot_switch=True

default_font_size=11
plt.rc('font', size=default_font_size) #controls default text size
plt.rc('axes', titlesize=default_font_size) #fontsize of the title
plt.rc('axes', labelsize=default_font_size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=default_font_size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=default_font_size) #fontsize of the y tick labels
plt.rc('legend', fontsize=default_font_size) #fontsize of the legend

#"Vienna" #"Pilsen" #"Clermont_Ferrand" #"Riga" "Bordeaux" "Grenoble" "Innsbruck" "Salzburg" "Kaunas"
#"VIE" #"PIL" #"CLF" #"RIG" "BOR" "GRE" "INN" "SAL" "KAU" "LIM" 
city1_string_in="Vienna"
city1_string_out="VIE" 
city2_string_in="Vienna"
city2_string_out="VIE" 

# city2_string_in="Salzburg"
# city2_string_out="SAL" 

# city2_string_in="Innsbruck"
# city2_string_out="INN" 


base_in_folder:  str ="Z:/NoiseML/2024/city_data_features/"
# base_out_folder: str ="Z:/NoiseML/2024/city_data_features/"
base_out_folder_pic: str ="Z:/NoiseML/2024/city_data_pics/"


# feature data

features = [
    "absorption",
    "dist2road",
    "dist2tree",
    "eu_dem_v11",
    "osm_streetinfo",
    "building_height"
]

# OSM street class, distance to clipped street class
in_grid_file1="_feat_dist2road.npy"
# EU DEM, divergernce from mean topography
in_grid_file2="_feat_eu_dem_v11.npy"
# Copernicus, TreeCoverDensity, distance to TCD
in_grid_file3="_feat_dist2tree.npy"
# OSM maximum speed and number of lanes, merged and smoothed, 0-1
in_grid_file4="_feat_osmmaxspeed_nolanes_smooth.npy"
# Urban atlas land use land cover, reclassified to represent absortpion, 0-10
in_grid_file5="_feat_absoprtion.npy"
# Urban atlas building height 0-x00 m
in_grid_file6="_feat_UA2012_bheight.npy"


# target noise data
in_grid_target="_target_noise_Aggroad_Lden.npy"

grid_c1_1=np.load(base_in_folder+city1_string_in+"/"+city1_string_out+in_grid_file1)
grid_c1_2=np.load(base_in_folder+city1_string_in+"/"+city1_string_out+in_grid_file2)
grid_c1_3=np.load(base_in_folder+city1_string_in+"/"+city1_string_out+in_grid_file3)
grid_c1_4=np.load(base_in_folder+city1_string_in+"/"+city1_string_out+in_grid_file4)
grid_c1_5=np.load(base_in_folder+city1_string_in+"/"+city1_string_out+in_grid_file5)
grid_c1_6=np.load(base_in_folder+city1_string_in+"/"+city1_string_out+in_grid_file6)
grid_c1_target=np.load(base_in_folder+city1_string_in+"/"+city1_string_out+in_grid_target)

grid_c2_1=np.load(base_in_folder+city2_string_in+"/"+city2_string_out+in_grid_file1)
grid_c2_2=np.load(base_in_folder+city2_string_in+"/"+city2_string_out+in_grid_file2)
grid_c2_3=np.load(base_in_folder+city2_string_in+"/"+city2_string_out+in_grid_file3)
grid_c2_4=np.load(base_in_folder+city2_string_in+"/"+city2_string_out+in_grid_file4)
grid_c2_5=np.load(base_in_folder+city2_string_in+"/"+city2_string_out+in_grid_file5)
grid_c2_6=np.load(base_in_folder+city2_string_in+"/"+city2_string_out+in_grid_file6)
grid_c2_target=np.load(base_in_folder+city2_string_in+"/"+city2_string_out+in_grid_target)

indexxy = np.where(grid_c1_1 == -999.25)
grid_c1_1[indexxy]=np.NAN
indexxy = np.where(grid_c1_2 == -999.25)
grid_c1_2[indexxy]=np.NAN
indexxy = np.where(grid_c1_3 == -999.25)
grid_c1_3[indexxy]=np.NAN
indexxy = np.where(grid_c1_4 == -999.25)
grid_c1_4[indexxy]=np.NAN
indexxy = np.where(grid_c1_5 == -999.25)
grid_c1_5[indexxy]=np.NAN
indexxy = np.where(grid_c1_6 == -999.25)
grid_c1_6[indexxy]=np.NAN

indexxy = np.where(grid_c2_1 == -999.25)
grid_c2_1[indexxy]=np.NAN
indexxy = np.where(grid_c2_2 == -999.25)
grid_c2_2[indexxy]=np.NAN
indexxy = np.where(grid_c2_3 == -999.25)
grid_c2_3[indexxy]=np.NAN
indexxy = np.where(grid_c2_4 == -999.25)
grid_c2_4[indexxy]=np.NAN
indexxy = np.where(grid_c2_5 == -999.25)
grid_c2_5[indexxy]=np.NAN
indexxy = np.where(grid_c2_6 == -999.25)
grid_c2_6[indexxy]=np.NAN

xc1_size=grid_c1_1.shape
xc2_size=grid_c2_1.shape

xc1_xstart=0
# xc1_xend=np.uint(xc1_size[0]/2)-1
xc1_xend=np.uint(xc1_size[0])
xc1_ystart=0
# xc1_yend=np.uint(xc1_size[1])
xc1_yend=np.uint(xc1_size[1]/2)

x_c1 = np.stack([grid_c1_1[xc1_xstart:xc1_xend,xc1_ystart:xc1_yend],
                 grid_c1_2[xc1_xstart:xc1_xend,xc1_ystart:xc1_yend],
                 grid_c1_3[xc1_xstart:xc1_xend,xc1_ystart:xc1_yend],
                 grid_c1_4[xc1_xstart:xc1_xend,xc1_ystart:xc1_yend],
                 grid_c1_5[xc1_xstart:xc1_xend,xc1_ystart:xc1_yend],
                 grid_c1_6[xc1_xstart:xc1_xend,xc1_ystart:xc1_yend]])

# xc2_xstart=np.uint(xc1_size[0]/2)+1
xc2_xstart=0
xc2_xend=np.uint(xc2_size[0])
# xc2_ystart=0
xc2_ystart=np.uint(xc1_size[1]/2)
xc2_yend=np.uint(xc2_size[1])

x_c2 = np.stack([grid_c2_1[xc2_xstart:xc2_xend,xc2_ystart:xc2_yend],
                 grid_c2_2[xc2_xstart:xc2_xend,xc2_ystart:xc2_yend],
                 grid_c2_3[xc2_xstart:xc2_xend,xc2_ystart:xc2_yend],
                 grid_c2_4[xc2_xstart:xc2_xend,xc2_ystart:xc2_yend],
                 grid_c2_5[xc2_xstart:xc2_xend,xc2_ystart:xc2_yend],
                 grid_c2_6[xc2_xstart:xc2_xend,xc2_ystart:xc2_yend]])


# x_c1=x_c1[:,0:]
#################################################
#  original data  
fig1, (axs1, axs2)  = plt.subplots(1, 2, figsize=(12, 8))

# im1=axs1.imshow(np.squeeze(x_c1[0,:,:]), cmap=colMap, vmin = 0.2)
# im2=axs2.imshow(np.squeeze(x_c2[0,:,:]), cmap=colMap, vmin = 0.2)

im1=axs1.imshow(np.squeeze(x_c1[0,:,:]))
im2=axs2.imshow(np.squeeze(x_c2[0,:,:]))

# x_dim=np.size(grid_c1_target,axis=0)
# y_dim=np.size(grid_c1_target,axis=1)
# y_line_slice=int(y_dim/2)
# plot1=axs1.plot([0,y_dim],[y_line_slice, y_line_slice],'-r')

plt.colorbar(im1, ax=axs1)
plt.colorbar(im2, ax=axs2)
# im1.set_clim(0,1)
# im2.set_clim(0,1)

# Standard Scaling of features:
def standard_scale(data):
    means = np.nanmean(data, axis=(0, 1), keepdims=True)
    stds = np.nanstd(data, axis=(0, 1), keepdims=True)
    return (data - means) / stds


# scalers = {}
# for i in range(x_c1.shape[0]):
#     # scalers[i] = StandardScaler()
#     # data_flat = x_c1[i,:,:].reshape(-1, 1)
#     # data_flat_scaled= scalers[i].fit_transform(data_flat)    
#     # x_c1[i, :, :]= data_flat_scaled.reshape(x_c1[i,:,:].shape)
#     x_c1[i, :, :]= standard_scale(x_c1[i,:,:])
    
# # x_c1 = x_c1.reshape((x_c1.shape[1], x_c1.shape[2], x_c1.shape[0]))

# scalers2 = {}
# for i in range(x_c2.shape[0]):
#     # scalers[i] = StandardScaler()
#     # data_flat = x_c2[i,:,:].reshape(-1, 1)
#     # data_flat_scaled= scalers[i].fit_transform(data_flat)    
#     # x_c2[i, :, :] = data_flat_scaled.reshape(x_c2[i,:,:].shape)
#     x_c2[i, :, :]= standard_scale(x_c2[i,:,:])
    
# x_c2 = x_c2.reshape((x_c2.shape[1], x_c2.shape[2], x_c2.shape[0]))



def calculate_ssim_between_cities(city1_data, city2_data):
    """Calculate SSIM for each feature between two cities and return the scores."""
    
    # No resizing of city data dimensions but cropping to overlap
    min_shape = (min(city1_data.shape[1], city2_data.shape[1]), min(city1_data.shape[2], city2_data.shape[2]))
    
    # Calculate SSIM for each feature
    ssim_scores = {}
    # orb_scores = {}
    # orb = cv2.ORB_create()
    
    for i, feature in enumerate(features):
        
        score = ssim(np.squeeze(city1_data[i,:min_shape[0], :min_shape[1]]), 
                     np.squeeze(city2_data[i,:min_shape[0], :min_shape[1]]), 
                     data_range=city2_data[i,:min_shape[0], :min_shape[1]].max() - city2_data[i,:min_shape[0], :min_shape[1]].min())
        ssim_scores[feature] = score
        
        # image1 = np.squeeze(city1_data[i,:min_shape[0], :min_shape[1]])
        # image2 = np.squeeze(city2_data[i,:min_shape[0], :min_shape[1]])
        # # Convert single channel to 3 channels (if needed)
        # if len(image1.shape) == 2:  # Grayscale image
        #     gimage1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        # if len(image2.shape) == 2:  # Grayscale image
        #     gimage2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        
        # kp1, des1 = orb.detectAndCompute(gimage1, None)
        # kp2, des2 = orb.detectAndCompute(gimage2, None)
        
        # # Use BFMatcher (Brute Force Matcher)
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = bf.match(des1, des2)

        # # Sort matches by distance (lower is better)
        # matches = sorted(matches, key=lambda x: x.distance)
        
        # orb_scores[feature] = matches
    
    # Calculate average SSIM
    average_ssim = np.mean(list(ssim_scores.values()))
    ssim_scores["average"] = average_ssim
    
    # return ssim_scores,orb_scores,  min_shape
    return ssim_scores, min_shape

ssim_scores, min_shape = calculate_ssim_between_cities(np.nan_to_num(x_c1,nan=0), np.nan_to_num(x_c2, nan=0))
print(f"SSIM Scores between {city1_string_in} and {city2_string_in}")
file = open(f"{base_in_folder}{city1_string_out}-{city2_string_out}_SSIM_scores.txt", "a")
file.writelines(f"SSIM Scores between {city1_string_in} and {city2_string_in}:")
file.writelines("\n")
file.close()
for feature, score in ssim_scores.items():
    if feature == "average":
        print(f"Average SSIM: {score:.4f}")
        file = open(f"{base_in_folder}{city1_string_out}-{city2_string_out}_SSIM_scores.txt", "a")
        file.writelines(f"Average SSIM: {score:.4f}")
        file.writelines("\n")
        file.close()
    else:
        print(f"{feature}: {score:.4f}")
        file = open(f"{base_in_folder}{city1_string_out}-{city2_string_out}_SSIM_scores.txt", "a")
        file.writelines(f"{feature}: {score:.4f}")
        file.writelines("\n")
        file.close()
print("------")
file = open("SSIM_scores.txt", "a")
file.writelines("------")
file.writelines("\n")
file.close()

# for feature, score in ssim_scores.items():
#     if feature == "average":
#         print(f"Average SSIM: {score:.4f}")
#         file = open(f"{base_in_folder}{city1_string_out}-{city2_string_out}_SSIM_scores.txt", "a")
#         file.writelines(f"Average SSIM: {score:.4f}")
#         file.writelines("\n")
#         file.close()
#     else:
#         print(f"{feature}: {score:.4f}")
#         file = open(f"{base_in_folder}{city1_string_out}-{city2_string_out}_SSIM_scores.txt", "a")
#         file.writelines(f"{feature}: {score:.4f}")
#         file.writelines("\n")
#         file.close()

# Get the colormap and set the under and bad colors
colMap = plt.cm.get_cmap("gist_rainbow").copy()
colMap.set_under(color='white')



# #################################################
# #  scaled data  
# fig2, (axs1, axs2)  = plt.subplots(1, 2, figsize=(12, 8))

# # im1=axs1.imshow(np.squeeze(x_c1[0,:,:]), cmap=colMap, vmin = 0.2)
# # im2=axs2.imshow(np.squeeze(x_c2[0,:,:]), cmap=colMap, vmin = 0.2)

# im1=axs1.imshow(np.squeeze(x_c1[0,:min_shape[0], :min_shape[1]]))
# im2=axs2.imshow(np.squeeze(x_c2[0,:min_shape[0], :min_shape[1]]))

# # x_dim=np.size(grid_c1_target,axis=0)
# # y_dim=np.size(grid_c1_target,axis=1)
# # y_line_slice=int(y_dim/2)
# # plot1=axs1.plot([0,y_dim],[y_line_slice, y_line_slice],'-r')

# plt.colorbar(im1, ax=axs1)
# plt.colorbar(im2, ax=axs2)
# # im1.set_clim(-1,1)
# # im2.set_clim(-1,1)
# plt.show

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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


plt.close('all')
plot_switch=True

default_font_size=11
plt.rc('font', size=default_font_size) #controls default text size
plt.rc('axes', titlesize=default_font_size) #fontsize of the title
plt.rc('axes', labelsize=default_font_size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=default_font_size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=default_font_size) #fontsize of the y tick labels
plt.rc('legend', fontsize=default_font_size) #fontsize of the legend

city_string_in1  ="Riga"
city_string_out1 ="RIG"
city_string_in2  ="Pilsen"
city_string_out2 ="PIL"
city_string_in3  ="Clermont_Ferrand"
city_string_out3 ="CLF"
city_string_in4  ="Salzburg"
city_string_out4 ="SAL"
city_string_in5  ="Vienna"
city_string_out5 ="VIE"
city_string_in6  ="Bordeaux"
city_string_out6 ="BOR"
city_string_in7  ="Grenoble"
city_string_out7 ="GRE"
city_string_in8  ="Innsbruck"
city_string_out8 ="INN"

base_in_folder="/home/sjet/data/323_end_noise/"

# feature data
#output figure file 
out_file = "_panel_features"

# target noise data
in_grid_target="_target_noise_Aggroad_Lden.npy"

grid_target1=np.load(base_in_folder+city_string_in1+"/"+city_string_out1+in_grid_target)
grid_target2=np.load(base_in_folder+city_string_in2+"/"+city_string_out2+in_grid_target)
grid_target3=np.load(base_in_folder+city_string_in3+"/"+city_string_out3+in_grid_target)
grid_target4=np.load(base_in_folder+city_string_in4+"/"+city_string_out4+in_grid_target)
grid_target5=np.load(base_in_folder+city_string_in5+"/"+city_string_out5+in_grid_target)
grid_target6=np.load(base_in_folder+city_string_in6+"/"+city_string_out6+in_grid_target)
grid_target7=np.load(base_in_folder+city_string_in7+"/"+city_string_out7+in_grid_target)
grid_target8=np.load(base_in_folder+city_string_in8+"/"+city_string_out8+in_grid_target)


grid_target1[np.where(grid_target1 <0)]=0
grid_target2[np.where(grid_target2 <0)]=0
grid_target3[np.where(grid_target3 <0)]=0
grid_target4[np.where(grid_target4 <0)]=0
grid_target5[np.where(grid_target5 <0)]=0
grid_target6[np.where(grid_target6 <0)]=0
grid_target7[np.where(grid_target7 <0)]=0
grid_target8[np.where(grid_target8 <0)]=0

# Get the colormap and set the under and bad colors
colMap = plt.cm.get_cmap("gist_rainbow").copy()
colMap.set_under(color='white')

if plot_switch:
    
    ###########################################
    #############plot all grids, panel of all
    print("#### Plotting file")
    
    fig, axs = plt.subplots(2, 4, figsize=(15, 12))
    plt.rcParams['axes.grid'] = False

    bin_levels=[35,40,45,50,55,60,65,70,75,80,85]
    
    im1=axs[0,0].hist(grid_target1.flatten(order='C'), bins=bin_levels)
    im1=axs[0,1].hist(grid_target2.flatten(order='C'), bins=bin_levels)
    im1=axs[0,2].hist(grid_target3.flatten(order='C'), bins=bin_levels)
    im1=axs[0,3].hist(grid_target4.flatten(order='C'), bins=bin_levels)
    im1=axs[1,0].hist(grid_target5.flatten(order='C'), bins=bin_levels)
    im1=axs[1,1].hist(grid_target6.flatten(order='C'), bins=bin_levels)
    im1=axs[1,2].hist(grid_target7.flatten(order='C'), bins=bin_levels)
    im1=axs[1,3].hist(grid_target8.flatten(order='C'), bins=bin_levels)
        
    axs[0,0].set_title(city_string_in1)
    axs[0,1].set_title(city_string_in2)
    axs[0,2].set_title(city_string_in3)
    axs[0,3].set_title(city_string_in4)
    axs[1,0].set_title(city_string_in5)
    axs[1,1].set_title(city_string_in6)
    axs[1,2].set_title(city_string_in7)
    axs[1,3].set_title(city_string_in8)
  
  
    
    plt.show()
    # plt.savefig(base_out_folder+city_string_in+"/"+city_string_out+out_file+".png")

  
    print("#### Plotting file done \n")



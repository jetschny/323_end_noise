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
plot_switch=False
write_switch=True

in_grid_file1="bcn_distance2road_urbanatlas.npy"
in_grid_file2="bcn_distance2topo_dem.npy"
in_grid_file3="bcn_distance2trees_tcd.npy"
in_grid_file4="OSM_roads_bcn_streetclass_clip.npy"
# in_grid_file5="OSM_roads_bcn_nlanes_clip_smooth.npy"
in_grid_file5="OSM_roads_bcn_nlanes_clip_kde15.npy"
# in_grid_file6="OSM_roads_bcn_maxspeed_clip_smooth.npy"
in_grid_file6="OSM_roads_bcn_maxspeed_clip_kde15.npy"

in_grid_target="2017_isofones_total_dia_mapa_estrategic_soroll_bcn_clip.npy"
out_grid_file ="2017_isofones_total_dia_mapa_estrategic_soroll_bcn_clip_predRFR01.npy"

grid1=np.load(in_grid_file1)
grid2=np.load(in_grid_file2)
grid3=np.load(in_grid_file3)
grid4=np.load(in_grid_file4)
grid5=np.load(in_grid_file5)
grid6=np.load(in_grid_file6)

grid_target=np.load(in_grid_target)
grid_target= grid_target.astype(float)

# x = np.linspace(1, grid1.shape[1], grid1.shape[1])
# y = np.linspace(1, grid1.shape[0], grid1.shape[0])
# X, Y = np.meshgrid(x, y)


# values of 0 in grid_target represent not-existing/missing data
# all values in the features at the same indices will be blanked out
# indexxy = np.where(grid_target == 0)
# # grid_target[indexxy]=0
# grid1[indexxy]=0
# grid2[indexxy]=0
# grid3[indexxy]=0
# grid4[indexxy]=0
# grid5[indexxy]=0
# grid6[indexxy]=0

# Get the colormap and set the under and bad colors
colMap = plt.cm.get_cmap("gist_rainbow").copy()
colMap.set_under(color='white')

# Mask the bad data:
# grid1_masked = np.ma.array(grid1,mask=np.isnan(grid1))

# if plot_switch:
#     fig, axs = plt.subplots(1, 4, figsize=(12, 8))
    
#     levels = np.arange(0, 3.5, 0.5)
    
#     # con1=ax1.contourf(grid1,[20, 50, 70], cmap='RdGy')
#     # con2=ax2.contourf(grid2,[5000, 15000, 20000], cmap='RdGy')
#     im1=axs[0].imshow(grid1, cmap=colMap, vmin = 0.2)
#     im2=axs[1].imshow(grid2, cmap="seismic")
#     im3=axs[2].imshow(grid3, cmap=colMap, vmin = 0.2)
#     im4=axs[3].imshow(grid_target, cmap=colMap, vmin = 0.2)
    
#      # plt.axis('off')
#     # plt.contourf(grid1)
#     plt.colorbar(im1, ax=axs[0])
#     plt.colorbar(im2, ax=axs[1])
#     plt.colorbar(im3, ax=axs[2])
#     plt.colorbar(im4, ax=axs[3])
#     axs[0].set_aspect('equal', 'box')
#     axs[1].set_aspect('equal', 'box')
#     axs[2].set_aspect('equal', 'box')
#     axs[3].set_aspect('equal', 'box')
#     im1.set_clim(0.1,8)
#     im2.set_clim(-10,10)
#     im3.set_clim(0.1,15)
#     im4.set_clim(40,80)
    
#     axs[0].set_title('Distance to Road')
#     axs[1].set_title('Divergence from Topo')
#     axs[2].set_title('Mean Tree Density')
#     axs[3].set_title('Modeled Noise')
    
#     # plt.colorbar(con2, ax=ax2)
#     plt.show()
    
    
# all non-existing / missing data at values of grid_target==0 will be excluded
indexxy = np.where(grid_target >0)
# indexxy = np.where(grid_target)
# grid_target_flat=grid_target[indexxy].flatten()
#conversion from dB to linear Pascal scale
# grid_target_flat_pa=10**(grid_target_flat/20)*(2e-5)

# grid_target_flat=grid_target.flatten()
# grid1_flat=grid1.flatten()
# grid2_flat=grid2.flatten()
# grid3_flat=grid3.flatten()
# grid4_flat=grid4.flatten()
# grid5_flat=grid5.flatten()
# grid6_flat=grid6.flatten()

# grid1_flat=grid1[indexxy].flatten()
# grid2_flat=grid2[indexxy].flatten()
# grid3_flat=grid3[indexxy].flatten()
# grid4_flat=grid4[indexxy].flatten()
# grid5_flat=grid5[indexxy].flatten()
# grid6_flat=grid6[indexxy].flatten()

# grid1_flat_max=np.max(grid1_flat)
# grid2_flat_max=np.max(grid2_flat)
# grid3_flat_max=np.max(grid3_flat)

# grid_target_flat_max=np.max(grid_target_flat)

# grid1_flat=grid1_flat/grid1_flat_max
# grid2_flat=grid2_flat/grid2_flat_max
# grid3_flat=grid3_flat/grid3_flat_max

# grid_target_flat=grid_target_flat/grid_target_flat_max

# grid_target_flat_pa=10**(grid_target_flat/20)*(2e-5)

df = pd.DataFrame(np.array((grid_target[indexxy].flatten(),
                            grid1[indexxy].flatten(), 
                            grid2[indexxy].flatten(),
                            grid3[indexxy].flatten(), 
                            grid4[indexxy].flatten(), 
                            grid5[indexxy].flatten(),
                            grid6[indexxy].flatten())).transpose(), 
                  columns=["Noise","Dist2Road","DivTopo","MeanTreeDens",
                           "StreetClass","NLanes","SpeedLimit"])

if plot_switch:
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.show()
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    sns.distplot(grid_target[indexxy].flatten(), label="Noise in dB", ax=axs[0])
    sns.distplot(10**(grid_target[indexxy].flatten()/20)*(2e-5), label="Noise in Pascal", ax=axs[1])
    axs[0].legend()
    axs[1].legend()
    plt.show()
    
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    counter=0
    for column in ["Dist2Road","DivTopo","MeanTreeDens", "StreetClass","NLanes","SpeedLimit"]:
        # print("counter  ",counter)
        # print("counter [] ",[np.int((counter-np.mod(counter,3))/3),np.mod(counter,3)])
        sns.distplot(df[column], label=column, ax=axs[np.int((counter-np.mod(counter,3))/3),np.mod(counter,3)])
        axs[np.int((counter-np.mod(counter,3))/3), np.mod(counter,3)].legend()
        counter=counter+1
    plt.show()

  
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    for column in df:
        sns.kdeplot(df[column], shade=True, label=column)

    plt.legend()
    plt.show()


# fig, axs = plt.subplots(1, 1, figsize=(12, 8))
# sns.pairplot(df,hue='Noise', height=2.5)
# plt.show()

# x = df[['Dist2Road', 'DivTopo', 'MeanTreeDens',"NLanes","SpeedLimit"]]
# x = x.join(pd.get_dummies(df.StreetClass))

x = df[['Dist2Road','MeanTreeDens',"NLanes","SpeedLimit"]]


# x = df[['Dist2Road']]
# x = df[['MeanTreeDens']]
y = 10**(df['Noise']/20)*(2e-5)
# y = df['Noise']

test_size=0.3
x_train, x_test = np.split(x,[int((1-test_size) * len(x))])
y_train, y_test = np.split(y,[int((1-test_size) * len(y))])
indexxy_train=(indexxy[0][0:int((1-test_size) * len(x))],indexxy[1][0:int((1-test_size) * len(x))])
indexxy_test=(indexxy[0][int((1-test_size) * len(x))::],indexxy[1][int((1-test_size) * len(x))::])

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 41)

print("##### Multi-vriable Regression ML  \n")

mlr = LinearRegression()  
mlr.fit(x_train, y_train)

print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(x, mlr.coef_))

#Prediction of test set
y_pred_mlr= mlr.predict(x_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_mlr))

mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
# slr_diff.head()

meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared              : {:.2f}'.format(mlr.score(x,y)*100))
print('Mean Absolute Error    : {:.5f}'.format(meanAbErr))
print('Mean Square Error      : {:.5f}'.format(meanSqErr))
print('Root Mean Square Error : {:.5f}'.format(rootMeanSqErr))


print("\n##### Multi-vriable Regression ML done... \n")

print("##### RF-decision tree ML \n")
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 10, random_state = 42,verbose=2, n_jobs= -1, 
                           warm_start = True)


# Train the model on training data
rf.fit(x_train, y_train);

# Use the forest's predict method on the test data
y_pred_test_rf = rf.predict(x_test)# Calculate the absolute errors
y_pred_train_rf = rf.predict(x_train)# Calculate the absolute errors

errors_rf = abs(y_pred_test_rf - y_test)# Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors_rf), 3), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / y_test)# Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')

print("\n##### RF-decision tree ML done \n")

# # Saving feature names for later use
# feature_list = list(x.columns)

# # Get numerical feature importances
# importances = list(rf.feature_importances_)# List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]# Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)# Print out the feature and importances 
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

meanAbErr = metrics.mean_absolute_error(y_test, y_pred_test_rf)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_test_rf)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test_rf))
print('R squared              : {:.2f}'.format(rf.score(x,y)*100))
print('Mean Absolute Error    : {:.5f}'.format(meanAbErr))
print('Mean Square Error      : {:.5f}'.format(meanSqErr))
print('Root Mean Square Error : {:.5f}'.format(rootMeanSqErr))

fig, axs = plt.subplots(2, 1, figsize=(12, 8))
axs[0].plot(range(1000), y_test[0:1000], "b-", range(1000), y_pred_test_rf[0:1000], "r--", range(1000), y_pred_mlr[0:1000], "g--")
axs[0].legend(["test", "RF prediction", "MLR predictions"])
axs[1].plot(range(1000), errors_rf[0:1000], "r-",range(1000), abs(y_pred_mlr - y_test)[0:1000], "g-" )
axs[1].legend(["RF prediction", "MLR predictions"])
plt.show()

if write_switch:
    print("#### Saving to npy file")
    grid_target_export=np.empty(grid_target.shape)*0
    
    grid_target_export[indexxy_train]=y_pred_train_rf
    grid_target_export[indexxy_test] =y_pred_test_rf
    np.save(out_grid_file,grid_target_export)
    print("#### Saving to npy file done")
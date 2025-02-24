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
import seaborn as sns
import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
import xgboost as xgb
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, confusion_matrix

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

#"Vienna" #"Pilsen" #"Clermont_Ferrand" #"Riga" "Bordeaux" "Grenoble" "Innsbruck" "Salzburg" "Kaunas"
#"VIE" #"PIL" #"CLF" #"RIG" "BOR" "GRE" "INN" "SAL" "KAU" "LIM" 
city1_string_in="Vienna"
city1_string_out="VIE" 
# city2_string_in="Vienna"
# city2_string_out="VIE" 

# city2_string_in="Salzburg"
# city2_string_out="SAL" 

# city2_string_in="Innsbruck"
# city2_string_out="INN" 

city2_string_in="Clermont_Ferrand"
city2_string_out="CLF" 


base_in_folder:  str ="Z:/NoiseML/2024/city_data_features/"
base_out_folder: str ="Z:/NoiseML/2024/city_data_MLpredictions/"
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

# output prediction noise data
out_grid_file1="predict_xgboost.npy"

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


indexxy = np.where(grid_c1_target == -999.25)
grid_c1_target[indexxy]=np.NAN
indexxy = np.where(grid_c2_target == -999.25)
grid_c2_target[indexxy]=np.NAN

noise_classes_old=sorted(np.unique(grid_c1_target))
noise_classes_old=noise_classes_old[:-1]
noise_classes_new=np.array(range(0, len(noise_classes_old), 1))

counter=0
for a in noise_classes_old:
    indexxy = np.where(grid_c1_target ==a)
    grid_c1_target[indexxy]=noise_classes_new[counter]
    counter=counter+1
    
counter=0
for a in noise_classes_old:
    indexxy = np.where(grid_c2_target ==a)
    grid_c2_target[indexxy]=noise_classes_new[counter]
    counter=counter+1
    
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

# xc1_size=grid_c1_1.shape
# xc2_size=grid_c2_1.shape

# xc1_xstart=0
# # xc1_xend=np.uint(xc1_size[0]/2)-1
# xc1_xend=np.uint(xc1_size[0])
# xc1_ystart=0
# xc1_yend=np.uint(xc1_size[1])
# # xc1_yend=np.uint(xc1_size[1]/2)

# x_c1 = np.stack([grid_c1_1[xc1_xstart:xc1_xend,xc1_ystart:xc1_yend],
#                  grid_c1_2[xc1_xstart:xc1_xend,xc1_ystart:xc1_yend],
#                  grid_c1_3[xc1_xstart:xc1_xend,xc1_ystart:xc1_yend],
#                  grid_c1_4[xc1_xstart:xc1_xend,xc1_ystart:xc1_yend],
#                  grid_c1_5[xc1_xstart:xc1_xend,xc1_ystart:xc1_yend],
#                  grid_c1_6[xc1_xstart:xc1_xend,xc1_ystart:xc1_yend]])

# # xc1_xstart=np.uint(xc1_size[0]/2)+1
# xc2_xstart=0
# xc2_xend=np.uint(xc2_size[0])
# xc2_ystart=0
# # xc1_ystart=np.uint(xc1_size[1]/2)
# xc2_yend=np.uint(xc2_size[1])

# x_c2 = np.stack([grid_c2_1[xc2_xstart:xc2_xend,xc2_ystart:xc2_yend],
#                  grid_c2_2[xc2_xstart:xc2_xend,xc2_ystart:xc2_yend],
#                  grid_c2_3[xc2_xstart:xc2_xend,xc2_ystart:xc2_yend],
#                  grid_c2_4[xc2_xstart:xc2_xend,xc2_ystart:xc2_yend],
#                  grid_c2_5[xc2_xstart:xc2_xend,xc2_ystart:xc2_yend],
#                  grid_c2_6[xc2_xstart:xc2_xend,xc2_ystart:xc2_yend]])


# x_c1=x_c1[:,0:]
#################################################
#  original data  
fig1, (axs1, axs2)  = plt.subplots(1, 2, figsize=(12, 8))

# im1=axs1.imshow(np.squeeze(x_c1[0,:,:]), cmap=colMap, vmin = 0.2)
# im2=axs2.imshow(np.squeeze(x_c2[0,:,:]), cmap=colMap, vmin = 0.2)

im1=axs1.imshow(np.squeeze(grid_c1_4[:,:]))
im2=axs2.imshow(np.squeeze(grid_c2_4[:,:]))

# x_dim=np.size(grid_c1_target,axis=0)
# y_dim=np.size(grid_c1_target,axis=1)
# y_line_slice=int(y_dim/2)
# plot1=axs1.plot([0,y_dim],[y_line_slice, y_line_slice],'-r')

plt.colorbar(im1, ax=axs1)
plt.colorbar(im2, ax=axs2)
im1.set_clim(0,1)
im2.set_clim(0,1)


# all non-existing / missing data at values of grid_target==0 will be excluded
indexxy = np.where(grid_c1_target >=0)
df1 = pd.DataFrame(np.array((grid_c1_target[indexxy].flatten(order='C'),
                            grid_c1_1[indexxy].flatten(order='C'), 
                            grid_c1_2[indexxy].flatten(order='C'),
                            grid_c1_3[indexxy].flatten(order='C'), 
                            grid_c1_4[indexxy].flatten(order='C'), 
                            grid_c1_5[indexxy].flatten(order='C'),
                            grid_c1_6[indexxy].flatten(order='C'),
                            indexxy[0], indexxy[1])).transpose(), 
                  columns=["Noise","Dist2Road","DivTopo","MeanTreeDens",
                            "StreetInfo","Absorption","BuildingHeight", "ArrayRowIndex", "ArrayColIndex"])

# all non-existing / missing data at values of grid_target==0 will be excluded
indexxy = np.where(grid_c2_target >=0)
df2 = pd.DataFrame(np.array((grid_c2_target[indexxy].flatten(order='C'),
                            grid_c2_1[indexxy].flatten(order='C'), 
                            grid_c2_2[indexxy].flatten(order='C'),
                            grid_c2_3[indexxy].flatten(order='C'), 
                            grid_c2_4[indexxy].flatten(order='C'), 
                            grid_c2_5[indexxy].flatten(order='C'),
                            grid_c2_6[indexxy].flatten(order='C'),
                            indexxy[0], indexxy[1])).transpose(), 
                  columns=["Noise","Dist2Road","DivTopo","MeanTreeDens",
                            "StreetInfo","Absorption","BuildingHeight", "ArrayRowIndex", "ArrayColIndex"])

                  
# df1_notnull = df1.query("Dist2Road.notnull() & DivTopo.notnull() & MeanTreeDens.notnull() & StreetInfo.notnull() & Absorption.notnull() & BuildingHeight.notnull()")
# df2_notnull = df2.query("Dist2Road.notnull() & DivTopo.notnull() & MeanTreeDens.notnull() & StreetInfo.notnull() & Absorption.notnull() & BuildingHeight.notnull()")

df1_notnull=df1
df2_notnull=df2

# Split data into features and target
X_c1 = df1_notnull.drop(["Noise"], axis = 1)
# X_c1 = df1_notnull[['Dist2Road','StreetInfo', "ArrayRowIndex", "ArrayColIndex"]]
# X_c1 = df1_notnull[["ArrayRowIndex", "ArrayColIndex"]]
y_c1 = df1_notnull['Noise']

# Split data into features and target
X_c2 = df2_notnull.drop(["Noise"], axis = 1)
# X_c2 = df2_notnull[['Dist2Road','StreetInfo', "ArrayRowIndex", "ArrayColIndex"]]
# X_c2 = df2_notnull[["ArrayRowIndex", "ArrayColIndex"]]
y_c2 = df2_notnull['Noise']


# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
test_size=0.3

x_c1_train, x_c1_test = np.split(X_c1,[int((1-test_size) * len(X_c1))])
y_c1_train, y_c1_test = np.split(y_c1,[int((1-test_size) * len(y_c1))])
# indexxy_train=(indexxy[0][0:int((1-test_size) * len(X_c1))],indexxy[1][0:int((1-test_size) * len(X_c1))])
# indexxy_test=(indexxy[0][int((1-test_size) * len(X_c1))::],indexxy[1][int((1-test_size) * len(X_c1))::])

indexxy_c1_train=(X_c1["ArrayRowIndex"][0:int((1-test_size) * len(X_c1))].to_numpy(dtype=int),
                        X_c1["ArrayColIndex"][0:int((1-test_size) * len(X_c1))].to_numpy(dtype=int) )
indexxy_c1_test=(X_c1["ArrayRowIndex"][int((1-test_size) * len(X_c1))::].to_numpy(dtype=int),
              X_c1["ArrayColIndex"][int((1-test_size) * len(X_c1))::].to_numpy(dtype=int) )

indexxy_c2=(X_c2["ArrayRowIndex"][:].to_numpy(dtype=int),
              X_c2["ArrayColIndex"][:].to_numpy(dtype=int) )


x_c1_train = x_c1_train.drop(["ArrayRowIndex", "ArrayColIndex"], axis=1)
x_c1_test = x_c1_test.drop(["ArrayRowIndex", "ArrayColIndex"], axis=1)
X_c2 = X_c2.drop(["ArrayRowIndex", "ArrayColIndex"], axis=1)


print("\n \n ##### XGBoost starting  \n")

# Convert the datasets into DMatrix format (required by XGBoost)
dtrain = xgb.DMatrix(x_c1_train, label=y_c1_train)
dtest = xgb.DMatrix(x_c1_test, label=y_c1_test)
dc2 = xgb.DMatrix(X_c2, label=y_c2)

# Specify parameters for XGBoost
params = {
    'objective': 'multi:softmax',  # For multi-class classification
    'num_class': 5,  # Number of classes
    "eval_metric": "mlogloss",  # Log loss for multi-class problems
    "max_depth": 8,             # Tree depth (try values from 3-10)
    "learning_rate": 0.05,       # Step size shrinkage
    "n_estimators": 500,        # Number of trees (boosting rounds)
    "subsample": 0.8,           # Fraction of training instances used for each tree
    "colsample_bytree": 0.8,    # Fraction of features used per tree
    "gamma": 0,                 # Minimum loss reduction for split (set > 0 to regularize)
    "lambda": 1,                # L2 regularization term
    "alpha": 0,                 # L1 regularization term (increase for feature selection)
    "tree_method": "hist"        # "hist" or "gpu_hist" for faster training on large datasets
}
    
# Train the model
num_round = 50  # Number of boosting rounds
bst = xgb.train(params, dtrain, num_round)

importance_score=bst.get_score(importance_type="gain")
print({key: val for key, val in sorted(importance_score.items(), key = lambda ele: ele[1])})

# Make predictions on the test set
y_c1_train_pred = bst.predict(dtrain)
y_c1_test_pred = bst.predict(dtest)
y_c2_pred = bst.predict(dc2)

# Evaluate the model using R² for both training and testing
r2_train = r2_score(y_c1_train, y_c1_train_pred)
r2_test = r2_score(y_c1_test, y_c1_test_pred)
r2_c2 = r2_score(y_c2, y_c2_pred)

# Print R² values for training and testing
print(f'R² (Training): {r2_train}')
print(f'R² (Testing): {r2_test}')
print(f'R² ({city2_string_in}): {r2_c2}')

# Calculate RMSE for both training and testing
rmse_train = root_mean_squared_error(y_c1_train, y_c1_train_pred)
rmse_test = root_mean_squared_error(y_c1_test, y_c1_test_pred)
rmse_c2 = root_mean_squared_error(y_c2, y_c2_pred)

print(f'RMSE (Training): {rmse_train}')
print(f'RMSE (Testing): {rmse_test}')
print(f'RMSE ({city2_string_in}): {rmse_c2}')

# Evaluate model accuracy
accuracy_test = accuracy_score(y_c1_test, y_c1_test_pred)
accuracy_train = accuracy_score(y_c1_train, y_c1_train_pred)
accuracy_c2 = accuracy_score(y_c2, y_c2_pred)
print(f'Accuracy (Training): {accuracy_train}')
print(f'Accuracy (Test): {accuracy_test}')
print(f'Accuracy ({city2_string_in}): {accuracy_c2}')

print("\n##### XGBoost prediction done... \n")



if plot_switch:
   
    #training on cityA and apply to cityA

    # Get and reshape confusion matrix data
    matrix = confusion_matrix(y_c1_test, y_c1_test_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)
    
    # Add labels to the plot
    # noise_classes_new=np.array(range(42, 80, 5))
    # noise_classes_new=np.append([32],noise_classes_new)
    # noise_classes_new=np.append(noise_classes_new,[87])
    
    class_names= [str(X) for X in noise_classes_old]
    # class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 
    #                'Cottonwood/Willow', 'Aspen', 'Douglas-fir',    
    #                'Krummholz']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=0, ha='left')
    plt.yticks(tick_marks2, class_names, rotation=0 )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f"Confusion Matrix for XGBoost Model trained on {city1_string_in} and infered to {city1_string_in}")
    plt.show()
    
    #training on cityA and apply to cityA

    # Get and reshape confusion matrix data
    matrix = confusion_matrix(y_c2, y_c2_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)
    
    # Add labels to the plot
    # noise_classes_new=np.array(range(42, 80, 5))
    # noise_classes_new=np.append([32],noise_classes_new)
    # noise_classes_new=np.append(noise_classes_new,[87])
    
    class_names= [str(X) for X in noise_classes_old]
    # class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 
    #                'Cottonwood/Willow', 'Aspen', 'Douglas-fir',    
    #                'Krummholz']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=0, ha='left')
    plt.yticks(tick_marks2, class_names, rotation=0 )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f"Confusion Matrix for XGBoost Model trained on {city1_string_in} and infered to {city2_string_in}")
    plt.show()
    
if write_switch:
    print("#### Saving to npy file")
    grid_c1_target_export=np.zeros(grid_c1_target.shape)-999.25
    
    grid_c1_target_export[indexxy_c1_train]=y_c1_train_pred
    grid_c1_target_export[indexxy_c1_test] =y_c1_test_pred
    np.save(f"{base_out_folder}{city1_string_in}/{city1_string_out}model_{city1_string_out}{out_grid_file1}",grid_c1_target_export)
    
    
    grid_c2_target_export=np.zeros(grid_c2_target.shape)-999.25
    grid_c2_target_export[indexxy_c2]=y_c2_pred
    
    np.save(f"{base_out_folder}{city2_string_in}/{city1_string_out}model_{city2_string_out}{out_grid_file1}",grid_c2_target_export)
    
    print("#### Saving to npy file done")
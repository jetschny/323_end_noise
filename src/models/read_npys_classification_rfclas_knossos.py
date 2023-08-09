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
# from sklearn.model_selection import train_test_split
import joblib
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
# from keras.utils import to_categorical 


plt.close('all')
plot_switch=True
write_switch=True
load_MLmod=False

default_font_size=11
plt.rc('font', size=default_font_size) #controls default text size
plt.rc('axes', titlesize=default_font_size) #fontsize of the title
plt.rc('axes', labelsize=default_font_size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=default_font_size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=default_font_size) #fontsize of the y tick labels
plt.rc('legend', fontsize=default_font_size) #fontsize of the legend

city_string_in="riga"
city_string_out="RIG"

base_in_folder="/home/sjet/repos/323_end_noise/data/processed/"
base_out_folder="/home/sjet/repos/323_end_noise/data/processed/"

# feature data
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

#output figure file 
out_file = "_panel_features"


in_model_file="models/2017_isofones_total_predRFC01_maxd10_compressed.joblib"
out_model_file="models/2017_isofones_total_predRFC01_maxd10_compressed.joblib"

grid1=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file1)
grid2=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file2)
grid3=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file3)
grid4=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file4)
grid5=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file5)
grid6=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file6)

grid_target=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_target)
# grid_target= grid_target.astype(float)

noise_classes_old=sorted(np.unique(grid_target))
noise_classes_new=np.array(range(0, len(noise_classes_old), 1))

counter=0
for a in noise_classes_old:
    indexxy = np.where(grid_target ==a)
    grid_target[indexxy]=noise_classes_new[counter]
    counter=counter+1
    

# Get the colormap and set the under and bad colors
colMap = plt.cm.get_cmap("gist_rainbow").copy()
colMap.set_under(color='white')


    
# all non-existing / missing data at values of grid_target==0 will be excluded
indexxy = np.where(grid_target >0)
df = pd.DataFrame(np.array((grid_target[indexxy].flatten(order='C'),
                            grid1[indexxy].flatten(order='C'), 
                            grid2[indexxy].flatten(order='C'),
                            grid3[indexxy].flatten(order='C'), 
                            grid4[indexxy].flatten(order='C'), 
                            grid5[indexxy].flatten(order='C'),
                            grid6[indexxy].flatten(order='C'))).transpose(), 
                  columns=["Noise","Dist2Road","DivTopo","MeanTreeDens",
                            "StreetInfo","Absortion","BuildingHeight"])

                  
x = df.drop(["Noise"], axis = 1)

# x = df[["Dist2Road","DivTopo","MeanTreeDens","StreetClass","NLanes","SpeedLimit","BuildingHeight"]]
# x = df[['Dist2Road','MeanTreeDens',"SpeedLimit"]]
# x = df[['Dist2Road',"SpeedLimit"]]
# x = df[['NLanes']]
# x = df['Dist2Road'][0::2]
# y = 10**(df['Noise']/20)*(2e-5)
y = df['Noise']
# y_cat = to_categorical(df["StreetClass"])

test_size=0.3

x_train, x_test = np.split(x,[int((1-test_size) * len(x))])
y_train, y_test = np.split(y,[int((1-test_size) * len(y))])
indexxy_train=(indexxy[0][0:int((1-test_size) * len(x))],indexxy[1][0:int((1-test_size) * len(x))])
indexxy_test=(indexxy[0][int((1-test_size) * len(x))::],indexxy[1][int((1-test_size) * len(x))::])

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 42)

print("\n \n ##### random forest classification ML  \n")

# fit the model on the whole dataset
if load_MLmod:   
    rfc_model = joblib.load(base_out_folder+in_model_file)
else:
    #Create a Gaussian Classifier
    rfc_model=RandomForestClassifier(n_estimators=10, random_state = 42,verbose=2, n_jobs= 6, 
                               warm_start = True, max_features="sqrt", max_depth=15)
    rfc_model.fit(x_train, y_train)
    joblib.dump(rfc_model, base_out_folder+out_model_file, compress=3)  # compression is ON!


# predict the class label
y_pred_test_rfc = rfc_model.predict(x_test)
y_pred_train_rfc= rfc_model.predict(x_train)
# summarize the predicted class
# print('Predicted Class: %d' % y_pred_logres[0])

# y_pred_train_rfc = y_train
# y_pred_test_rfc = np.ones(len(x_test))

print("\n\n Random forest classification Train Accuracy : {:.5f} \n\n ".format(metrics.accuracy_score(y_train, y_pred_train_rfc)))
print("\n\n Random forest classification Test Accuracy  : {:.5f} \n\n ".format(metrics.accuracy_score(y_test, y_pred_test_rfc)))

# print(metrics.confusion_matrix(y_test, y_pred_rfc))

if plot_switch:
    f_i = list(zip(list(x.columns),rfc_model.feature_importances_))
    f_i.sort(key = lambda x : x[1])
    plt.barh([x[0] for x in f_i],[x[1] for x in f_i])

    plt.show()

    # Get and reshape confusion matrix data
    matrix = metrics.confusion_matrix(y_test, y_pred_test_rfc)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)
    
    # Add labels to the plot
    noise_classes_new=np.array(range(42, 80, 5))
    noise_classes_new=np.append([32],noise_classes_new)
    noise_classes_new=np.append(noise_classes_new,[87])
    
    class_names= [str(x) for x in noise_classes_new]
    # class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 
    #                'Cottonwood/Willow', 'Aspen', 'Douglas-fir',    
    #                'Krummholz']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=0, ha='left')
    plt.yticks(tick_marks2, class_names, rotation=0 )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.show()
    
    matrix2=np.copy(matrix)
    for ii in range(1,9):
        for jj in range(0,9):
            matrix2[ii,jj]=matrix[ii,jj-1]+matrix[ii,jj]+matrix[ii,jj+1]
    
    matrix2[0,0]=matrix[0,0]+matrix[0,1]
    matrix2[9,9]=matrix[9,9]+matrix[9,8]
    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix2, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)
    
    # Add labels to the plot
    noise_classes_new=np.array(range(42, 80, 5))
    noise_classes_new=np.append([32],noise_classes_new)
    noise_classes_new=np.append(noise_classes_new,[87])
    
    class_names= [str(x) for x in noise_classes_new]
    # class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 
    #                'Cottonwood/Willow', 'Aspen', 'Douglas-fir',    
    #                'Krummholz']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=0, ha='left')
    plt.yticks(tick_marks2, class_names, rotation=0 )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.show()

# meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
# meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
# rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
# print('R squared              : {:.2f}'.format(mlr.score(x,y)*100))
# print('Mean Absolute Error    : {:.5f}'.format(meanAbErr))
# print('Mean Square Error      : {:.5f}'.format(meanSqErr))
# print('Root Mean Square Error : {:.5f}'.format(rootMeanSqErr))

print("\n##### random forest classification ML  done... \n")

if write_switch:
    print("#### Saving to npy file")
    grid_target_export=np.zeros(grid_target.shape)
    
    grid_target_export[indexxy_train]=y_pred_train_rfc
    grid_target_export[indexxy_test] =y_pred_test_rfc
    np.save(base_out_folder+out_grid_file,grid_target_export)
    # np.save(base_in_folder+out_grid_file,grid_target)
    print("#### Saving to npy file done")

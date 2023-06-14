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
write_switch=False
load_MLmod=False

plt.rc('font', size=12) #controls default text size
plt.rc('axes', titlesize=12) #fontsize of the title
plt.rc('axes', labelsize=12) #fontsize of the x and y labels
plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
plt.rc('ytick', labelsize=12) #fontsize of the y tick labels
plt.rc('legend', fontsize=12) #fontsize of the legend

base_in_folder="/home/sjet/data/323_end_noise/BCN_data/"
base_out_folder="/home/sjet/data/323_end_noise/BCN_data/"

in_grid_file1="MES2017_Transit_Lden_3035_clip_predRFC01_1dB_re5dB_BCN.npy"
in_grid_file2="MES2017_Transit_Lden_3035_clip_1db_re5dB.npy"


grid_pred=np.load(base_in_folder+in_grid_file1)
grid_target=np.load(base_in_folder+in_grid_file2)

# noise_classes_old=sorted(np.unique(grid_target))
# noise_classes_new=np.array(range(0, len(noise_classes_old), 1))

# counter=0
# for a in noise_classes_old:
#     indexxy = np.where(grid_target ==a)
#     grid_target[indexxy]=noise_classes_new[counter]
#     counter=counter+1
    
# Get the colormap and set the under and bad colors
colMap = plt.cm.get_cmap("gist_rainbow").copy()
colMap.set_under(color='white')

indexxy = np.where(grid_target >0)

print("\n\n Random forest classification Train Accuracy : {:.5f} \n\n ".format(metrics.accuracy_score(grid_target[indexxy].flatten(order='C'), grid_pred[indexxy].flatten(order='C'))))
# print(metrics.confusion_matrix(y_test, y_pred_rfc))

if plot_switch:
    # f_i = list(zip(list(x.columns),rfc_model.feature_importances_))
    # f_i.sort(key = lambda x : x[1])
    # plt.barh([x[0] for x in f_i],[x[1] for x in f_i])

    # plt.show()

    y_test=grid_target[indexxy].flatten(order='C')
    y_pred=grid_pred[indexxy].flatten(order='C')
    
    # Get and reshape confusion matrix data
    matrix = metrics.confusion_matrix(y_test, y_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2,  vmin=0, vmax=0.6)
    
    # Add labels to the plot
    noise_classes_mat=np.array(range(42, 80, 5))
    noise_classes_mat=np.append([32],noise_classes_mat)
    noise_classes_mat=np.append(noise_classes_mat,[87])
    
    class_names= [str(x) for x in noise_classes_mat]
    # class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 
    #                'Cottonwood/Willow', 'Aspen', 'Douglas-fir',    
    #                'Krummholz']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=0, ha='left')
    plt.yticks(tick_marks2, class_names, rotation=0 )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model 1dB input')
    
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
    # grid_target_export=np.empty(grid_target.shape)*0
    
    # grid_target_export[indexxy_train]=y_pred_train_rfc
    # grid_target_export[indexxy_test] =y_pred_test_rfc
    # np.save(base_out_folder+out_grid_file,grid_target_export)
    print("#### Saving to npy file done")

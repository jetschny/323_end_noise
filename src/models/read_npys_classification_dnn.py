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
from sklearn import metrics
# from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# import tensorflow as tf
from tensorflow.keras.utils import to_categorical 
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflowkeras.optimizers import SGD, Adam, Adadelta, RMSprop
# import keras.backend as K
from sklearn.metrics import classification_report
import joblib

plt.close('all')
plot_switch=True
write_switch=True
load_MLmod=False


plt.rc('font', size=12) #controls default text size
plt.rc('axes', titlesize=12) #fontsize of the title
plt.rc('axes', labelsize=12) #fontsize of the x and y labels
plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
plt.rc('ytick', labelsize=12) #fontsize of the y tick labels
plt.rc('legend', fontsize=12) #fontsize of the legend

base_in_folder="BCN_data/"

in_grid_file1="bcn_dist2road_urbanatlas_osm_merge.npy"
in_grid_file2="bcn_distance2topo_dem.npy"
in_grid_file3="bcn_distance2trees_tcd.npy"
in_grid_file4="OSM_roads_bcn_streetclass_clip.npy"
# in_grid_file5="OSM_roads_bcn_nlanes_clip_smooth.npy"
in_grid_file5="OSM_roads_bcn_nlanes_clipfill_kde15.npy"
# in_grid_file6="OSM_roads_bcn_maxspeed_clip_smooth.npy"
in_grid_file6="OSM_roads_bcn_maxspeed_clipfill_kde15.npy"
#in_grid_file7="bcn_road_focalstats50_clip.npy"


in_grid_target="2017_isofones_total_dia_mapa_estrategic_soroll_bcn_clip.npy"
out_grid_file ="bcn_noise_predDNN_01.npy"
in_model_file="bcn_noise_predDNN_01.joblib"
out_model_file="cn_noise_predDNN_01.joblib"

grid1=np.load(base_in_folder+in_grid_file1)
grid2=np.load(base_in_folder+in_grid_file2)
grid3=np.load(base_in_folder+in_grid_file3)
grid4=np.load(base_in_folder+in_grid_file4)
grid5=np.load(base_in_folder+in_grid_file5)
grid6=np.load(base_in_folder+in_grid_file6)

grid_target=np.load(base_in_folder+in_grid_target)
grid_target= grid_target.astype(float)

noise_classes_old=sorted(np.unique(grid_target))
noise_classes_new=np.array(range(0, len(noise_classes_old), 1))

counter=0
for a in noise_classes_old:
    indexxy = np.where(grid_target ==a)
    grid_target[indexxy]=noise_classes_new[counter]
    counter=counter+1
    

# indexxy = np.where((grid_target <55) & (grid_target>0))
# grid_target[indexxy]=1
# indexxy = np.where(grid_target >55)
# grid_target[indexxy]=2

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
                            grid6[indexxy].flatten(order='C'), )).transpose(), 
                  columns=["Noise","Dist2Road","DivTopo","MeanTreeDens",
                            "StreetClass","NLanes","SpeedLimit"])

# Remove all null values from position:
# Remove Missing Values 
# df= df[df['Noise'] != 0]
df = df[(df[["Noise","Dist2Road","DivTopo","MeanTreeDens","SpeedLimit"]] != 0).all(axis=1)]


x = df.drop(["Noise", "StreetClass"], axis = 1)
# x = df.drop("Noise", axis = 1)
# x = df[['Dist2Road','MeanTreeDens',"NLanes","SpeedLimit"]][0::2]
# x = df[['Dist2Road',"SpeedLimit"]]
# x = df['Dist2Road']
# x=df.drop(columns=["Noise"])
# x = df[['SpeedLimit']]
# x = df[['NLanes']]
# x = df['MeanTreeDens']
# y = 10**(df['Noise']/20)*(2e-5)
y = df['Noise']-1


# Standard Scaling of features:
sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(x))

# Converting prediction to categorical:

y_cat = to_categorical(y-min(y))


test_size=0.3
# convert integers to dummy variables (i.e. one hot encoded)
# x_train, x_test, y_train, y_test = train_test_split(x.values, y, test_size = test_size, random_state = 42)


x_train, x_test = np.split(x,[int((1-test_size) * len(x))])
y_train, y_test = np.split(y_cat,[int((1-test_size) * len(y))])
indexxy_train=(indexxy[0][0:int((1-test_size) * len(x))],indexxy[1][0:int((1-test_size) * len(x))])
indexxy_test=(indexxy[0][int((1-test_size) * len(x))::],indexxy[1][int((1-test_size) * len(x))::])



print("\n \n ##### dnn classification ML  \n")

# define the keras model
model = Sequential()
model.add(Dense(10, input_shape = (2,), activation = "relu"))
model.add(Dense(5, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation = "softmax"))

# define the keras model
# model = Sequential()
# model.add(Dense(12, input_shape=(2,), activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


# compile the keras model
# model.compile(optimizer=Adam(lr = 0.01), loss="categorical_crossentropy", metrics = ["accuracy"])
# model.compile(optimizer='Adam', loss="binary_crossentropy", metrics = ["accuracy"])
# model.summary()


# fit the model on the whole dataset
if load_MLmod:   
    model = joblib.load(base_in_folder+in_model_file)
else:
    #Create a Gaussian Classifier
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=10)
    joblib.dump(model, base_in_folder+out_model_file, compress=3)  # compression is ON!
    


# predict the class label
# y_pred_dnn = model.predict(x_test)
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)
# y_pred = classifier.predict(x_test)

# summarize the predicted class
# print('Predicted Class: %d' % y_pred_logres[0])

y_pred_train_class = np.argmax(y_pred_train, axis=1)
y_pred_test_class = np.argmax(y_pred_test, axis=1)
y_test_class = np.argmax(y_test, axis=1)
# y_test_class = y_test


# print("DNN classification Train Accuracy : {:.5f}".format(metrics.accuracy_score(y_train, model.predict(x_train))))
# print("DNN forest classification Test Accuracy  : {:.5f}".format(metrics.accuracy_score(y_test, model.predict(x_test))))

# print(metrics.confusion_matrix(y_test, y_pred_rfc))

if plot_switch:
    # f_i = list(zip(list(x.columns),rfc_model.feature_importances_))
    # f_i.sort(key = lambda x : x[1])
    # plt.barh([x[0] for x in f_i],[x[1] for x in f_i])

    plt.show()

    # Get and reshape confusion matrix data
    matrix = metrics.confusion_matrix(y_test_class, y_pred_test_class)
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
        matrix2[ii,ii]=matrix2[ii,ii-1]+matrix2[ii,ii]+matrix2[ii,ii+1]
    
    matrix2[0,0]=matrix2[0,0]+matrix2[0,1]
    matrix2[9,9]=matrix2[9,9]+matrix2[9,8]
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

print("\n##### dnn classification ML  done... \n")


# fig, axs = plt.subplots(2, 1, figsize=(12, 8))
# axs[0].plot(range(1000), y_test[0:1000], "b-", range(1000), y_pred_rf[0:1000], "r--", range(1000), y_pred_mlr[0:1000], "g--")
# axs[0].legend(["test", "RF prediction", "MLR predictions"])
# axs[1].plot(range(1000), errors_rf[0:1000], "r-",range(1000), abs(y_pred_mlr - y_test)[0:1000], "g-" )
# axs[1].legend(["RF prediction", "MLR predictions"])
# plt.show()

print(classification_report(y_test_class, y_pred_test_class))

if write_switch:
    print("#### Saving to npy file")
    grid_target_export=np.empty(grid_target.shape)*0
    
    grid_target_export[indexxy_train]=y_pred_train_class
    grid_target_export[indexxy_test] =y_pred_test_class
    np.save(base_in_folder+out_grid_file,grid_target_export)
    print("#### Saving to npy file done")
  
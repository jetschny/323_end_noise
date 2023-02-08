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
# import seaborn as sns
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


plt.close('all')
plot_switch=False


plt.rc('font', size=12) #controls default text size
plt.rc('axes', titlesize=12) #fontsize of the title
plt.rc('axes', labelsize=12) #fontsize of the x and y labels
plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
plt.rc('ytick', labelsize=12) #fontsize of the y tick labels
plt.rc('legend', fontsize=12) #fontsize of the legend

in_grid_file1="bcn_distance2road_urbanatlas.npy"
in_grid_file2="bcn_distance2topo_dem.npy"
in_grid_file3="bcn_distance2trees_tcd.npy"
in_grid_file4="OSM_roads_bcn_streetclass_clip.npy"
# in_grid_file5="OSM_roads_bcn_nlanes_clip_smooth.npy"
in_grid_file5="OSM_roads_bcn_nlanes_clip_kde15.npy"
# in_grid_file6="OSM_roads_bcn_maxspeed_clip_smooth.npy"
in_grid_file6="OSM_roads_bcn_maxspeed_clip_kde15.npy"

in_grid_target="2017_isofones_total_dia_mapa_estrategic_soroll_bcn_clip.npy"

grid1=np.load(in_grid_file1)
grid2=np.load(in_grid_file2)
grid3=np.load(in_grid_file3)
grid4=np.load(in_grid_file4)
grid5=np.load(in_grid_file5)
grid6=np.load(in_grid_file6)

grid_target=np.load(in_grid_target)
grid_target= grid_target.astype(float)


# Get the colormap and set the under and bad colors
colMap = plt.cm.get_cmap("gist_rainbow").copy()
colMap.set_under(color='white')


    
# all non-existing / missing data at values of grid_target==0 will be excluded
indexxy = np.where(grid_target >0)

df = pd.DataFrame(np.array((grid_target[indexxy].flatten(),
                            grid1[indexxy].flatten(), 
                            grid2[indexxy].flatten(),
                            grid3[indexxy].flatten(), 
                            grid4[indexxy].flatten(), 
                            grid5[indexxy].flatten(),
                            grid6[indexxy].flatten())).transpose(), 
                  columns=["Noise","Dist2Road","DivTopo","MeanTreeDens",
                           "StreetClass","NLanes","SpeedLimit"])


x = df[['Dist2Road','MeanTreeDens',"NLanes","SpeedLimit"]]
# x = df[['Dist2Road']]
# x = df[['MeanTreeDens']]
# y = 10**(df['Noise']/20)*(2e-5)
y = df['Noise']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 41)

print("\n \n ##### logisticial  Regression ML  \n")

# logreg = LogisticRegression()
# rfe = RFE(logreg, step=30)


# rfe = rfe.fit(x_train, y_train.values.ravel())
# print(rfe.support_)
# print(rfe.ranking_)

logres_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# define the model evaluation procedure
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
# n_scores = cross_val_score(logres_model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
# print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# fit the model on the whole dataset
logres_model.fit(x_train, y_train)
# predict the class label
y_pred_logres = logres_model.predict(x_test)
# summarize the predicted class
# print('Predicted Class: %d' % y_pred_logres[0])

print("Multinomial Logistic regression Train Accuracy : {:.5f}".format(metrics.accuracy_score(y_train, logres_model.predict(x_train))))
print("Multinomial Logistic regression Test Accuracy  : {:.5f}".format(metrics.accuracy_score(y_test, logres_model.predict(x_test))))
    
# mlr = LinearRegression()  
# mlr.fit(x_train, y_train)

# print("Intercept: ", mlr.intercept_)
# print("Coefficients:")
# list(zip(x, mlr.coef_))

# #Prediction of test set
# y_pred_mlr= mlr.predict(x_test)
# #Predicted values
# print("Prediction for test set: {}".format(y_pred_mlr))

# mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
# # slr_diff.head()

# meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
# meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
# rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
# print('R squared              : {:.2f}'.format(mlr.score(x,y)*100))
# print('Mean Absolute Error    : {:.5f}'.format(meanAbErr))
# print('Mean Square Error      : {:.5f}'.format(meanSqErr))
# print('Root Mean Square Error : {:.5f}'.format(rootMeanSqErr))

print("\n##### logisticial  Regression ML  done... \n")


# fig, axs = plt.subplots(2, 1, figsize=(12, 8))
# axs[0].plot(range(1000), y_test[0:1000], "b-", range(1000), y_pred_rf[0:1000], "r--", range(1000), y_pred_mlr[0:1000], "g--")
# axs[0].legend(["test", "RF prediction", "MLR predictions"])
# axs[1].plot(range(1000), errors_rf[0:1000], "r-",range(1000), abs(y_pred_mlr - y_test)[0:1000], "g-" )
# axs[1].legend(["RF prediction", "MLR predictions"])
# plt.show()

  
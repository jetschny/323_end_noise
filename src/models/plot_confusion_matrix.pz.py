# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 09:31:09 2025

@author: jetschny
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
# Define your confusion matrix Vienna
confusion_matrix = np.array([
    [0.7905, 0.1859, 0.0202, 0.0032, 0.0002],
    [0.3863, 0.4822, 0.1067, 0.0208, 0.0040],
    [0.1434, 0.4172, 0.3265, 0.0978, 0.0151],
    [0.0702, 0.2306, 0.3638, 0.2983, 0.0370],
    [0.0349, 0.0950, 0.2210, 0.3753, 0.2737]
])

# Define your confusion matrix CLF
confusion_matrix = np.array([
    [0.6797, 0.2883, 0.0302, 0.0017, 0.0001],
    [0.1482, 0.5634, 0.2603, 0.0270, 0.0010],
    [0.0161, 0.2248, 0.5037, 0.2335, 0.0218],
    [0.0051, 0.0411, 0.2831, 0.5083, 0.1624],
    [0.0056, 0.0168, 0.0462, 0.2448, 0.6866]
    ])
 
# Class labels (optional, customize as needed)
# labels = ['Class 55 dB', 'Class 60 dB', 'Class 65 dB', 'Class 70 dB', 'Class 75 DB']
labels = ['55 dB', '60 dB', '65 dB', '70 dB', '75 DB']
 
# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt=".4f", cmap="Blues", xticklabels=labels, yticklabels=labels)
 
# Customization
plt.title('Confusion Matrix for Vienna')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
 
# Save as an image file
plt.tight_layout()
plt.savefig('clf_confusion_matrix.png', dpi=300)
 
# Display the plot
plt.show()
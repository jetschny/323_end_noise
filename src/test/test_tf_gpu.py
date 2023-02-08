#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 22:01:58 2022

@author: sjet
"""
import tensorflow as tf 

if tf.test.gpu_device_name(): 

    print('Default GPU Device:    {}'.format(tf.test.gpu_device_name()))

else:

    print("Please install GPU version of TF")
   

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)
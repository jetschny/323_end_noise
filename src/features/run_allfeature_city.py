# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:12:23 2024

@author: jetschny
"""

import os
import subprocess
list_city_names=["Athens", "Bordeaux", "Budapest", "Clermont_Ferrand", "Debrecen", "Grenoble", "Innsbruck", "Kaunas",
                 "Limassol", "Ljubljana", "Madrid", "Maribor", "Nicosia", "Oslo", "Pilsen", "Riga", "Salzburg", "Thessaloniki", "Vienna"]

list_city_names=["Athens"]

# list_scripts=["tiffraster2npy_noise_knossos.py",
#               "tiffraster2npy_absorption_knossos.py",
#               "tiffraster2npy_dem_knossos.py",
#               "tiffraster2npy_height_knossos.py",
#               "tiffraster2npy_LocalClimateZones_knossos.py",
#               "tiffraster2npy_tcd_knossos.py",
#               "load_osm_data_streets_knossos.py",
#               "read_npy_distance2roads_knossos.py",
#               "read_npy_osmsmooth_knossos.py"]

# list_scripts=["tiffraster2npy_noise_knossos.py",
#               "tiffraster2npy_absorption_knossos.py",
#               "tiffraster2npy_dem_knossos.py",
#               "tiffraster2npy_height_knossos.py",
#               "tiffraster2npy_tcd_knossos.py"]


# list_scripts=["load_osm_data_streets_knossos.py",
               # "read_npy_distance2roads_knossos.py",
               # "read_npy_osmsmooth_knossos.py"]

# list_scripts=[ "read_npy_osmsmooth_knossos.py"]
list_scripts=[ "../visualization/npy2tiffraster_dnnpredict_knossos.py"]

for ii in list_city_names:
    city_name=ii
    city_initials=ii[:3].upper()
    for jj in list_scripts:
        results=""
    
        result=subprocess.run(["python",jj,city_name,city_initials,"False","True"],stdout=subprocess.PIPE, text=True, encoding="cp437")
        print(result.stdout)


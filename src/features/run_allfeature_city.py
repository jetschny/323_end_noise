# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:12:23 2024

@author: jetschny
"""

import os
import subprocess
city_name="Bordeaux"
city_initials="BOR"

# os.system("python file1.py", shell=True)

# os.system("python tiffraster2npy_dem_knossos.py Clermont_Ferrand CLF True True")
# os.system("python tiffraster2npy_dem_knossos.py Grenoble GRE True True")
# os.system("python tiffraster2npy_dem_knossos.py Innsbruck INN True True")
# os.system("python tiffraster2npy_dem_knossos.py Kaunas KAU True True")
# os.system("python tiffraster2npy_dem_knossos.py Limassol LIM True True")
# os.system("python tiffraster2npy_dem_knossos.py Madrid MAD True True")
# os.system("python tiffraster2npy_dem_knossos.py Nicosia NIC True True")
# os.system("python tiffraster2npy_dem_knossos.py Pilsen PIL True True")
# os.system("python tiffraster2npy_dem_knossos.py Riga RIG True True")
# os.system("python tiffraster2npy_dem_knossos.py Salzburg SAL True True")
# os.system("python tiffraster2npy_dem_knossos.py Vienna VIE True True")

# target data
# os.system("python tiffraster2npy_noise_knossos.py Bordeaux BOR True True")

list_scripts=["tiffraster2npy_noise_knossos.py",
              "tiffraster2npy_absorption_knossos.py",
              "tiffraster2npy_dem_knossos.py",
              "tiffraster2npy_height_knossos.py",
              "tiffraster2npy_tcd_knossos.py",
              "load_osm_data_streets_knossos.py"
              "read_npy_distance2roads_knossos.py",
              "read_npy_osmsmooth_knossos.py"]

# list_scripts=["tiffraster2npy_noise_knossos.py",
#               "tiffraster2npy_absorption_knossos.py"]

for ii in list_scripts:
    results=""
    result=subprocess.run(["python",ii,city_name,city_initials,"True","True"],stdout=subprocess.PIPE, text=True, encoding="cp437")
    print(result.stdout)

# # raster feature data
# os.system("python tiffraster2npy_absorption_knossos.py Bordeaux BOR True True")
# os.system("python tiffraster2npy_dem_knossos.py Bordeaux BOR True True")
# os.system("python tiffraster2npy_height_knossos.py Bordeaux BOR True True")
# os.system("python tiffraster2npy_tcd_knossos.py Bordeaux BOR True True")

# # vector OSM data
# os.system("python load_osm_data_streets_knossos.py Bordeaux BOR True True")
# os.system("python read_npy_distance2roads_knossos.py Bordeaux BOR True True")
# os.system("python read_npy_osmsmooth_knossos.py Bordeaux BOR True True")
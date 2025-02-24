import numpy as np
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler

base_in_folder:  str ="P:/NoiseML/2024/city_data_features/"

def load_np_data(city):
    city = city
    '''
    in_grid_files = [base_in_folder+"/clermont_ferrand/VIE_feat_absoprtion.npy",
                 base_in_folder+"/clermont_ferrand/VIE_feat_dist2road.npy",
                 base_in_folder+"/clermont_ferrand/VIE_feat_dist2tree.npy",
                 base_in_folder+"/clermont_ferrand/VIE_feat_eu_dem_v11.npy",
                 base_in_folder+"/clermont_ferrand/VIE_feat_osmmaxspeed_nolanes_smooth.npy",
                 base_in_folder+"/clermont_ferrand/VIE_feat_UA2012_bheight.npy"]

    for in_grid_file in in_grid_files:
        grid = np.load(in_grid_file)
        if np.any(np.isinf(grid)):
            print(f"Infinity values found in file: {in_grid_file}")
    '''
    if city == 'vienna':
        in_grid_file1=base_in_folder+"/Vienna/VIE_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/Vienna/VIE_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/Vienna/VIE_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/Vienna/VIE_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/Vienna/VIE_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/Vienna/VIE_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/Vienna/VIE_target_noise_Aggroad_Lden.npy"
    
    elif city == 'clf':
        in_grid_file1=base_in_folder+"/clermont_ferrand/CLF_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/clermont_ferrand/CLF_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/clermont_ferrand/CLF_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/clermont_ferrand/CLF_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/clermont_ferrand/CLF_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/clermont_ferrand/CLF_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/clermont_ferrand/CLF_target_noise_Aggroad_Lden.npy"
    
    elif city == 'riga':
        in_grid_file1=base_in_folder+"/riga/RIG_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/riga/RIG_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/riga/RIG_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/riga/RIG_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/riga/RIG_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/riga/RIG_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/riga/RIG_target_noise_Aggroad_Lden.npy"
        
    elif city == 'budapest':
        in_grid_file1=base_in_folder+"/budapest/BUD_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/budapest/BUD_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/budapest/BUD_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/budapest/BUD_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/budapest/BUD_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/budapest/BUD_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/budapest/BUD_target_noise_Aggroad_Lden.npy"
    
    elif city == 'pilsen':
        in_grid_file1=base_in_folder+"/pilsen/PIL_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/pilsen/PIL_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/pilsen/PIL_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/pilsen/PIL_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/pilsen/PIL_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/pilsen/PIL_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/pilsen/PIL_target_noise_Aggroad_Lden.npy"
        
    elif city == 'grenoble':
        in_grid_file1=base_in_folder+"/grenoble/GRE_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/grenoble/GRE_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/grenoble/GRE_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/grenoble/GRE_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/grenoble/GRE_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/grenoble/GRE_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/grenoble/GRE_target_noise_Aggroad_Lden.npy"
        
    elif city == 'innsbruck':
        in_grid_file1=base_in_folder+"/innsbruck/INN_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/innsbruck/INN_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/innsbruck/INN_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/innsbruck/INN_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/innsbruck/INN_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/innsbruck/INN_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/innsbruck/INN_target_noise_Aggroad_Lden.npy"
        
    elif city == 'salzburg':
        in_grid_file1=base_in_folder+"/salzburg/SAL_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/salzburg/SAL_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/salzburg/SAL_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/salzburg/SAL_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/salzburg/SAL_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/salzburg/SAL_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/salzburg/SAL_target_noise_Aggroad_Lden.npy"
        
    elif city == 'nicosia':
        in_grid_file1=base_in_folder+"/nicosia/NIC_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/nicosia/NIC_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/nicosia/NIC_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/nicosia/NIC_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/nicosia/NIC_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/nicosia/NIC_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/nicosia/NIC_target_noise_Aggroad_Lden.npy"
        
    elif city == 'kaunas':
        in_grid_file1=base_in_folder+"/kaunas/KAU_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/kaunas/KAU_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/kaunas/KAU_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/kaunas/KAU_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/kaunas/KAU_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/kaunas/KAU_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/kaunas/KAU_target_noise_Aggroad_Lden.npy"
     
    elif city == 'bordeaux':
        in_grid_file1=base_in_folder+"/bordeaux/BOR_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/bordeaux/BOR_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/bordeaux/BOR_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/bordeaux/BOR_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/bordeaux/BOR_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/bordeaux/BOR_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/bordeaux/BOR_target_noise_Aggroad_Lden.npy"
    
    elif city == 'limassol':
        in_grid_file1=base_in_folder+"/limassol/LIM_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/limassol/LIM_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/limassol/LIM_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/limassol/LIM_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/limassol/LIM_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/limassol/LIM_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/limassol/LIM_target_noise_Aggroad_Lden.npy"
        
    elif city == 'madrid':
        in_grid_file1=base_in_folder+"/madrid/MAD_feat_absoprtion.npy"
        in_grid_file2=base_in_folder+"/madrid/MAD_feat_dist2road.npy"
        in_grid_file3=base_in_folder+"/madrid/MAD_feat_dist2tree.npy"
        in_grid_file4=base_in_folder+"/madrid/MAD_feat_eu_dem_v11.npy"
        in_grid_file5=base_in_folder+"/madrid/MAD_feat_osmmaxspeed_nolanes_smooth.npy"
        in_grid_file6=base_in_folder+"/madrid/MAD_feat_UA2012_bheight.npy"
        in_grid_targe=base_in_folder+"/madrid/MAD_target_noise_Aggroad_Lden.npy"
    
    else:
        print("Invalid city, Choose from the following: vienna, clf, riga, pilsen, innsbruck, salzburg, grenoble.")
    
    grid1=np.load(in_grid_file1)
    grid2=np.load(in_grid_file2)
    grid3=np.load(in_grid_file3)
    grid4=np.load(in_grid_file4)
    grid5=np.load(in_grid_file5)
    grid6=np.load(in_grid_file6)
    
    grid_target=np.load(in_grid_targe)
    grid_target= grid_target.astype(float)
    
    noise_classes_old=sorted(np.unique(grid_target))
    noise_classes_new=np.array(range(0, len(noise_classes_old), 1))

    counter=0
    for a in noise_classes_old:
        indexxy = np.where(grid_target ==a)
        grid_target[indexxy]=noise_classes_new[counter]
        counter=counter+1
    
    indexxy_null = np.where(grid1 == -999.25)
    grid1[indexxy_null]=np.NZERO
    indexxy_null = np.where(grid2 == -999.25)
    grid2[indexxy_null]=np.NZERO
    indexxy_null = np.where(grid3 == -999.25)
    grid3[indexxy_null]=np.NZERO
    indexxy_null = np.where(grid4 == -999.25)
    grid4[indexxy_null]=np.NZERO
    indexxy_null = np.where(grid5 == -999.25)
    grid5[indexxy_null]=np.NZERO
    indexxy_null = np.where(grid6 == -999.25)
    grid6[indexxy_null]=np.NZERO
    
    x = np.stack([grid1,grid2,grid3,grid4,grid5,grid6])
    
    # Standard Scaling of features:
    scalers = {}
    for i in range(x.shape[0]):
        scalers[i] = StandardScaler()
        x[i, :, :] = scalers[i].fit_transform(x[i, :, :])    
    x = x.reshape((x.shape[1], x.shape[2], x.shape[0]))
        
    return x


features = [
    "absorption",
    "dist2road",
    "dist2tree",
    "eu_dem_v11",
    "osm_maxspeed",
    "building_height"
]
# city_data_dict = {
#     "Pilsen": load_np_data('pilsen'),  
#     "Clf": load_np_data('clf'),        
#     "Riga": load_np_data('riga'),      
#     "Vienna": load_np_data('vienna'),
#     "Grenoble": load_np_data('grenoble'),
#     "Innsbruck": load_np_data('innsbruck'),
#     "Salzburg": load_np_data('salzburg'),
#     "Nicosia": load_np_data('nicosia'),
#     "Budapest": load_np_data('budapest'),
#     #"Bordeaux":load_np_data('bordeaux'),
#     "Kaunas": load_np_data('kaunas'),
#     "Limassol": load_np_data('limassol'),
#     "Madrid": load_np_data('madrid')
# }

# city_data_dict = {
#     "Clf": load_np_data('clf'),        
#     "Vienna": load_np_data('vienna'),
#     "Innsbruck": load_np_data('innsbruck'),
#     "Salzburg": load_np_data('salzburg'),
# }

city_data_dict = {
    "Clf": load_np_data('clf'),        
    "Vienna": load_np_data('vienna'),
}


def resize_city_data(source_data, target_shape):
    """Resize source city data to match the target shape."""
    resized_data = np.empty((target_shape[0], target_shape[1], source_data.shape[2]))
    for i in range(source_data.shape[2]):  # Loop through each feature
        resized_data[:, :, i] = resize(source_data[:, :, i], (target_shape[0], target_shape[1]), anti_aliasing=True)
    return resized_data

def calculate_ssim_between_cities(city1_data, city2_data):
    """Calculate SSIM for each feature between two cities and return the scores."""
    # Resize the second city data to match the first city data dimensions
    city2_resampled = resize_city_data(city2_data, city1_data.shape[:2])
    
    # Calculate SSIM for each feature
    ssim_scores = {}
    for i, feature in enumerate(features):
        score = ssim(city1_data[:, :, i], city2_resampled[:, :, i], data_range=city2_resampled[:, :, i].max() - city2_resampled[:, :, i].min())
        ssim_scores[feature] = score
    
    # Calculate average SSIM
    average_ssim = np.mean(list(ssim_scores.values()))
    ssim_scores["average"] = average_ssim
    
    return ssim_scores

# Iterate through each pair of cities by their names and calculate SSIM
city_names = list(city_data_dict.keys())
for i in range(len(city_names)):
    for j in range(i + 1, len(city_names)):
        city1_name = city_names[i]
        city2_name = city_names[j]
        ssim_scores = calculate_ssim_between_cities(city_data_dict[city1_name], city_data_dict[city2_name])
        print(f"SSIM Scores between {city1_name} and {city2_name}:")
        file = open("SSIM_scores.txt", "a")
        file.writelines(f"SSIM Scores between {city1_name} and {city2_name}:")
        file.writelines("\n")
        file.close()
        for feature, score in ssim_scores.items():
            if feature == "average":
                print(f"Average SSIM: {score:.4f}")
                file = open("SSIM_scores.txt", "a")
                file.writelines(f"Average SSIM: {score:.4f}")
                file.writelines("\n")
                file.close()
            else:
                print(f"{feature}: {score:.4f}")
                file = open("SSIM_scores.txt", "a")
                file.writelines(f"{feature}: {score:.4f}")
                file.writelines("\n")
                file.close()
        print("------")
        file = open("SSIM_scores.txt", "a")
        file.writelines("------")
        file.writelines("\n")
        file.close()
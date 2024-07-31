####################################################################################
#create and save for samebasin intake files
#cityrivnum = f'{h08dir}/global_city/dat/riv_num_/city_basins.json'
#city_basins_list
####################################################################################
    
import os
import json
import numpy as np
import pandas as pd

####################################################################################
city_len = 1860
distance = 100
ex_flg = False
save_flag = False
####################################################################################

# paths
h08dir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08'
pop_path = f'{h08dir}/global_city/dat/pop_tot_/GPW4ag__20100000.gl5'
riv_path = f'{h08dir}/wsi/dat/riv_out_/W5E5LR__00000000.gl5'
rivnum_path = f'{h08dir}/global_city/dat/riv_num_/rivnum.CAMA.gl5'
rivnxl_path = f"{h08dir}/global_city/dat/riv_nxl_/rivnxl.CAMA.gl5"
msk_dir = f'{h08dir}/global_city/dat/vld_cty_'
cnt_dir = f'{h08dir}/global_city/dat/cty_cnt_/gpw4'
prf_dir = f'{h08dir}/global_city/dat/cty_prf_'

# save path
cityrivnum = f'{h08dir}/global_city/dat/riv_num_/city_basins_{distance}km.json'
basin_cities_path = f'{h08dir}/global_city/dat/riv_num_/basin_to_cities_{distance}km.json'

# intake
int_dir = f'{h08dir}/global_city/dat/cty_int_/{distance}km_samebasin'

# rivpath
rivnxl_gl5 = np.fromfile(rivnxl_path, 'float32').reshape(2160, 4320)
pop = np.fromfile(pop_path, dtype='float32').reshape(2160, 4320)
rivout = np.fromfile(riv_path, dtype='float32').reshape(2160, 4320)
rivnum_gl5 = np.fromfile(rivnum_path, dtype='float32').reshape(2160, 4320)

# lonlat
lonlat_path = f'{h08dir}/global_city/dat/cty_lst_/gpw4/WUP2018_300k_2010.txt'
with open(lonlat_path, "r") as input_file:
    lines = input_file.readlines()

####################################################################################
# JOB. 1.
####################################################################################
# identifying same rivnum cities into list?
city_basins = {}
for i in range(city_len):
    city_num = i+1
    ctymsk_path = f'{msk_dir}/city_{city_num:08}.gl5'
    prf_path = f'{prf_dir}/vld_cty_/city_{city_num:08}.gl5'
    int_path = f'{int_dir}/city_{city_num:08}.gl5'

    if not os.path.exists(prf_path):
        print(f'{city_num} is invalid prf')

    else:
        ctymsk = np.fromfile(ctymsk_path, dtype='float32').reshape(2160, 4320)
        prf = np.fromfile(prf_path, dtype='float32').reshape(2160, 4320)
        intake = np.fromfile(int_path, dtype='float32').reshape(2160, 4320)
        rivnum_mask = np.ma.masked_where((intake+prf) == 0, rivnum_gl5)
        rivnum_lst = np.unique(rivnum_mask)[:-1]
        city_basins[city_num] = rivnum_lst.filled(1e20)

        print(f'{city_num} done')

print(city_basins)

basin_to_cities = {}
for city, basins in city_basins.items():
    for rivnum in basins:
        if rivnum not in basin_to_cities:
            basin_to_cities[rivnum] = []
        basin_to_cities[rivnum].append(city)

city_basins_list = {city: basins.tolist() for city, basins in city_basins.items()}
cityrivnum = f'{h08dir}/global_city/dat/riv_num_/city_basins_{distance}km.json'

# save city_basins_list
if save_flag is True:
    with open(cityrivnum, 'w') as json_file:
        json.dump(city_basins_list, json_file)
    print(f'{cityrivnum} saved')
else:
    print(f'save_flag: {save_flag}, NOT SAVED')

####################################################################################
#create and save below files
#basin_cities_path = f'{h08dir}/global_city/dat/riv_num_/basin_to_cities.json'
#new_basin_to_cities
####################################################################################

####################################################################################
# JOB. 2.
####################################################################################

# cities at same basin
basin_to_cities = {}
for city, basins in city_basins.items():
    for rivnum in basins:
        if rivnum not in basin_to_cities:
            basin_to_cities[rivnum] = []
        basin_to_cities[rivnum].append(city)

for rivnum, cities in basin_to_cities.items():
    if len(cities) > 1:  # 複数の都市に共有されている流域番号のみ表示
        print(f"rivnum {rivnum}: {cities}")

new_basin_to_cities = {rivnum: cities for rivnum, cities in basin_to_cities.items() if len(cities) > 1}

# save new_basin_to_cities
if save_flag is True:
    with open(basin_cities_path, 'w') as json_file:
        json.dump(new_basin_to_cities, json_file)
        print(f'{basin_cities_path} saved')
else:
    print(f'save_flag: {save_flag}, NOT SAVED')

"""
Author Kajiyama @ 20240402
+ if no PRF due to basin number or no main river in downtown
+ if grid_num <= 1, no PRF
+ elif no main river in downtown, PRF=highest elevation
+ elif main river too short (<= 2), PRF=higher grid of main river
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------------------
# MODULES
#---------------------------------------------------------------------------------------------------------------

def l_coordinate_to_tuple(lcoordinate, a=2160, b=4320):
    lat_l = a - ((lcoordinate - 1) // b)
    lon_l = (lcoordinate) % b - 1
    return (lat_l, lon_l)


#---------------------------------------------------------------------------------------------------------------

def explore(target_index, remove_grid, innercity_grid, width, save_flag=False):
    """
    A: After over remove_grid process
    B: After over remove_grid process & over innercity_grid process
    """
    latgrd = 2160 # sum of latitude grids (y)
    longrd = 4320 # sum of longitude grids (x)

    # pop data
    POP = 'gpw4'

#---------------------------------------------------------------------------------------------------------------
#   PATH
#---------------------------------------------------------------------------------------------------------------

    # root directory
    root_dir = "/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city"
    # lonlat data
    city_path = f"{root_dir}/dat/cty_lst_/{POP}/WUP2018_300k_2010.txt"
    # city mask data
    cmsk_dir = f"{root_dir}/dat/cty_msk_/{POP}"
    # riv data
    rivnum_path = f"{root_dir}/dat/riv_num_/rivnum.CAMA.gl5"
    rivara_path = f"{root_dir}/dat/riv_ara_/rivara.CAMA.gl5"
    rivnxl_path = f"{root_dir}/dat/riv_nxl_/rivnxl.CAMA.gl5"

#---------------------------------------------------------------------------------------------------------------
#   City Lon Lat Information
#---------------------------------------------------------------------------------------------------------------

    """
    first line of all_lines
    1 35.6895 139.6917 36859.626 Tokyo """

    # city_list.txt
    with open(city_path, "r") as input_file:
        lines = input_file.readlines()

    line = lines[target_index-1]
    parts = line.split('\t') 
    parts = [item.strip() for item in parts]
    city_num = int(parts[0]) 
    cnt_lat = float(parts[1]) 
    cnt_lon = float(parts[2]) 
    city_name = parts[4].replace("\"", "").replace("?", "").replace("/", "") 

    lonmin = float(cnt_lon - width)
    lonmax = float(cnt_lon + width)
    latmin = float(cnt_lat - width)
    latmax = float(cnt_lat + width)

    print(f"city_num {city_num}")
    print(city_name)

#---------------------------------------------------------------------------------------------------------------
#   Get Lon Lat
#---------------------------------------------------------------------------------------------------------------

    """Calculate the indices corresponding to the desired latitudes and longitudes"""
    # West from UK is negative 0 <= lon <= -180
    # East from UK is positive 0 <= lon <= 180
    # small value to larger value (34-36, 138-140)
    lat_length = np.linspace(-90, 90, latgrd+1)
    lon_length = np.linspace(-180, 180, longrd+1)
    lat_start, lat_end = np.searchsorted(lat_length, [latmin, latmax])
    lon_start, lon_end = np.searchsorted(lon_length, [lonmin, lonmax])

    # adjust to 0.25 grid
    # lat
    if lat_start%3 == 0:
        lat_start = lat_start
    elif lat_start%3 == 1:
        lat_start -= 1
    elif lat_start%3 == 2:
        lat_start += 1
    # lon
    if lon_start%3 == 0:
        lon_start = lon_start
    elif lon_start%3 == 1:
        lon_start -= 1
    elif lon_start%3 == 2:
        lon_start += 1

    width_grid = width * 12 * 2

    lat_end = lat_start + width_grid
    lon_end = lon_start + width_grid

#---------------------------------------------------------------------------------------------------------------
#   Load city mask data (g_mask_cropped)
#---------------------------------------------------------------------------------------------------------------

    g_mask = np.fromfile(f'{cmsk_dir}/city_{city_num:08d}.gl5', 'float32').reshape(latgrd, longrd)
    g_mask = np.flipud(g_mask)
    g_mask = np.ma.masked_where(g_mask >= 1E20, g_mask)
    g_mask_cropped = g_mask[lat_start:lat_end, lon_start:lon_end]
    g_mask_cropped = np.flipud(g_mask_cropped)

#---------------------------------------------------------------------------------------------------------------
#   Load basin data (g_rivnum_cropped)
#---------------------------------------------------------------------------------------------------------------

    g_rivnum = np.fromfile(rivnum_path, 'float32').reshape(latgrd, longrd)
    g_rivnum = np.flipud(g_rivnum)
    g_rivnum = np.ma.masked_where(g_rivnum >= 1E20, g_rivnum)
    g_rivnum_cropped = g_rivnum[lat_start:lat_end, lon_start:lon_end]
    g_rivnum_cropped = np.flipud(g_rivnum_cropped)
    g_rivnum_cropped = np.ma.masked_where(~np.isfinite(g_rivnum_cropped) | (g_rivnum_cropped == 0), g_rivnum_cropped)

#---------------------------------------------------------------------------------------------------------------
#   Load upper river catchment area (g_rivara_cropped)
#---------------------------------------------------------------------------------------------------------------

    g_rivara = np.fromfile(rivara_path, 'float32').reshape(latgrd, longrd)
    g_rivara = np.flipud(g_rivara)
    g_rivara = np.ma.masked_where(g_rivara >= 1E20, g_rivara)
    g_rivara_cropped = g_rivara[lat_start:lat_end, lon_start:lon_end]
    g_rivara_cropped = np.flipud(g_rivara_cropped)
    g_rivara_cropped = np.ma.masked_where(~np.isfinite(g_rivara_cropped) | (g_rivara_cropped == 0), g_rivara_cropped)

#---------------------------------------------------------------------------------------------------------------
#   Load river's next l coordinate data (g_rivnxl_cropped)
#---------------------------------------------------------------------------------------------------------------

    g_rivnxl = np.fromfile(rivnxl_path, 'float32').reshape(latgrd, longrd)
    g_rivnxl = np.flipud(g_rivnxl)
    g_rivnxl = np.ma.masked_where(g_rivnxl >= 1E20, g_rivnxl)
    g_rivnxl_cropped = g_rivnxl[lat_start:lat_end, lon_start:lon_end]
    g_rivnxl_cropped = np.flipud(g_rivnxl_cropped)
    g_rivnxl_cropped = np.ma.masked_where(~np.isfinite(g_rivnxl_cropped) | (g_rivnxl_cropped == 0), g_rivnxl_cropped)

#---------------------------------------------------------------------------------------------------------------
#   Basin data only where city mask exists (g_rivnum_cropped_city)
#---------------------------------------------------------------------------------------------------------------

    g_rivnum_cropped_city = np.where(g_mask_cropped == 1, g_rivnum_cropped, np.nan)
    g_rivnum_cropped_city = np.ma.masked_where(~np.isfinite(g_rivnum_cropped_city) | (g_rivnum_cropped_city == 0), g_rivnum_cropped_city)

#---------------------------------------------------------------------------------------------------------------
#   3D array consists of g_rivara_cropped + g_rivnum_cropped (g_ara_num_cropped)
#---------------------------------------------------------------------------------------------------------------

    # g_ara_num_croppedを構造化配列として作成
    dtype = [('rivara', 'float32'), ('rivnum', 'float32')]
    g_ara_num_cropped = np.empty(g_rivara_cropped.shape, dtype=dtype)

    # rivaraとrivnumのデータをg_ara_num_croppedに追加
    g_ara_num_cropped['rivara'] = g_rivara_cropped
    g_ara_num_cropped['rivnum'] = g_rivnum_cropped

#---------------------------------------------------------------------------------------------------------------
#  Basin over remove_grid (Rivnum_A_array)
#---------------------------------------------------------------------------------------------------------------

    g_rivnum_cropped_masked = np.ma.masked_array(g_rivnum_cropped, np.isnan(g_rivnum_cropped))

    unique_values_org, counts_org = np.unique(g_rivnum_cropped_masked.compressed(), return_counts=True)
    value_counts_dict = dict(zip(unique_values_org, counts_org))

    sorted_dict_by_value_descending = dict(sorted(value_counts_dict.items(), key=lambda item: item[1], reverse=True))

    filtered_dict_A = {key: value for key, value in sorted_dict_by_value_descending.items() if value >= remove_grid}

    Rivnum_A_array = np.ma.masked_all(g_rivnum_cropped_masked.shape, dtype='float32')

    for rivnum_id in filtered_dict_A.keys():
        matching_positions = np.where(g_rivnum_cropped_masked.data == rivnum_id)
        Rivnum_A_array[matching_positions] = rivnum_id

    Rivnum_A_array = np.ma.masked_where(~np.isfinite(Rivnum_A_array) | (Rivnum_A_array == 0), Rivnum_A_array)

#---------------------------------------------------------------------------------------------------------------
#   Basin over remove_grid within city mask (Rivnum_A_array_citymasked)
#---------------------------------------------------------------------------------------------------------------

    invalid_mask = np.isnan(Rivnum_A_array) | (Rivnum_A_array == 0)
    Rivnum_A_array_citymasked = np.ma.masked_where((g_mask_cropped != 1) | invalid_mask, Rivnum_A_array)

#---------------------------------------------------------------------------------------------------------------
#   (unique_values_A)
#---------------------------------------------------------------------------------------------------------------

    unique_values_A, counts_A = np.unique(Rivnum_A_array_citymasked.compressed(), return_counts=True)
    value_counts_dict_A = dict(zip(unique_values_A, counts_A))

#---------------------------------------------------------------------------------------------------------------
#   (rivara_max_array_A)
#---------------------------------------------------------------------------------------------------------------

    # データ型とサイズに基づいて新しい配列を作成
    rivara_max_array_A = np.ma.masked_all(g_ara_num_cropped.shape, dtype='float32')

    for rivnum_id in value_counts_dict_A.keys():
        # 同じrivnumの位置を取得
        matching_positions = np.where(Rivnum_A_array_citymasked == rivnum_id)
        # これらの位置におけるrivaraの最大値の位置を取得
        max_rivara_position = np.argmax(g_rivara_cropped[matching_positions])
        # 最大のrivaraの位置に対応するrivnumを新しい配列に保存する
        # 河口グリッドに該当
        rivara_max_array_A[matching_positions[0][max_rivara_position], matching_positions[1][max_rivara_position]] = rivnum_id

#---------------------------------------------------------------------------------------------------------------
#   riv nxtl -> lonlat coordinate array 24x24x2 (riv_nxlonlat_cropped)
#---------------------------------------------------------------------------------------------------------------

    # l coordiate to lonlat coordinate
    vfunc = np.vectorize(l_coordinate_to_tuple, otypes=[tuple])
    riv_nxlonlat = np.empty(g_rivnxl_cropped.shape, dtype=tuple)
    mask = ~np.isnan(g_rivnxl_cropped)
    riv_nxlonlat[mask] = vfunc(g_rivnxl_cropped[mask])
    riv_nxlonlat_shape = (riv_nxlonlat.shape[0], riv_nxlonlat.shape[1], 2)

    riv_nxlonlat_lst = []
    for row in riv_nxlonlat:
        for x, y in row:
            # width_grid = cropped scale(24x24)
            modified_x = width_grid - (x - lat_start)
            modified_y = y - lon_start
            riv_nxlonlat_lst.append((modified_x, modified_y))

    riv_nxlonlat_cropped = np.array(riv_nxlonlat_lst).reshape(riv_nxlonlat_shape)
    riv_nxlonlat_cropped = riv_nxlonlat_cropped.astype(int)

#---------------------------------------------------------------------------------------------------------------
#   (path_dict)
#   (riv_path_array_A)
#---------------------------------------------------------------------------------------------------------------

    path_dict = {}
    riv_path_array_A = np.ma.masked_all(rivara_max_array_A.shape, dtype='float32')
    visited_coords = set()

    for uid in unique_values_A:
        coords_a = np.argwhere(rivara_max_array_A == uid)
        riv_path_array_A[coords_a[0][0], coords_a[0][1]] = uid
        if coords_a.size > 0:
            target_coord = tuple(coords_a[0]) 
            path_coords = [target_coord]
            for _ in range(len(g_mask_cropped)):
                if target_coord in visited_coords:
                    break
                visited_coords.add(target_coord)
                matched_coords = np.argwhere(np.all(target_coord == riv_nxlonlat_cropped, axis=2))
                if len(matched_coords) == 0:
                    break
                unvisited_matched = [tuple(coord) for coord in matched_coords if tuple(coord) not in visited_coords]
                if not unvisited_matched:
                    break
                rivara_values = [g_rivara_cropped[coord[0], coord[1]] for coord in unvisited_matched]
                max_index = np.argmax(rivara_values)
                best_coord = unvisited_matched[max_index]
                riv_path_array_A[best_coord[0], best_coord[1]] = uid
                target_coord = best_coord 
                path_coords.append(target_coord)

            path_dict[uid] = path_coords

#---------------------------------------------------------------------------------------------------------------
#   Rivpath over innercity_grid (riv_path_city_B)
#---------------------------------------------------------------------------------------------------------------

    # city mask
    fill_value = 1e20
    riv_path_array_filled = riv_path_array_A.filled(fill_value)
    riv_path_city_A = np.where(g_mask_cropped==1, riv_path_array_filled, fill_value)

    # make new array
    riv_path_city_B = copy.deepcopy(riv_path_city_A)

    for uid in unique_values_A:
        count = 0
        mask = (riv_path_city_A == uid)
        count = np.sum(mask)
        if count < innercity_grid:
            riv_path_city_B[riv_path_city_B== uid] = fill_value

    riv_path_city_B = np.ma.masked_where(riv_path_city_B >= fill_value, riv_path_city_B)

#---------------------------------------------------------------------------------------------------------------
#   Update unique river basin number after 2 removing process (unique_values_B)
#---------------------------------------------------------------------------------------------------------------

    unique_values_B, _ = np.unique(riv_path_city_B.compressed(), return_counts=True)

#---------------------------------------------------------------------------------------------------------------
#   (Rivnum_B_array)
#---------------------------------------------------------------------------------------------------------------

    Rivnum_B_array = np.ma.masked_all(g_rivnum_cropped_masked.shape, dtype='float32')

    for uid in unique_values_B:
        row_indices, col_indices = np.where(Rivnum_A_array == uid)
        Rivnum_B_array[row_indices, col_indices] = uid

#---------------------------------------------------------------------------------------------------------------
#   Updated river mouse grid (rivara_max_array_B)
#---------------------------------------------------------------------------------------------------------------

    rivara_max_array_B = np.ma.masked_all(g_ara_num_cropped.shape, dtype='float32')

    for rivnum_id in unique_values_B:
        matching_positions = np.where(Rivnum_A_array_citymasked == rivnum_id)
        max_rivara_position = np.argmax(g_rivara_cropped[matching_positions])
        rivara_max_array_B[matching_positions[0][max_rivara_position], matching_positions[1][max_rivara_position]] = rivnum_id

#---------------------------------------------------------------------------------------------------------------
#   Update riv_path_array with full length out of city mask (riv_path_array_B)
#---------------------------------------------------------------------------------------------------------------

    path_dict = {}
    riv_path_array_B = np.ma.masked_all(rivara_max_array_B.shape, dtype='float32')
    visited_coords = set()

    for uid in unique_values_B:
        coords_a = np.argwhere(rivara_max_array_B == uid)
        riv_path_array_B[coords_a[0][0], coords_a[0][1]] = uid
        if coords_a.size > 0:
            target_coord = tuple(coords_a[0]) 
            path_coords = [target_coord]
            for _ in range(len(g_mask_cropped)):
                if target_coord in visited_coords:
                    break
                visited_coords.add(target_coord)
                matched_coords = np.argwhere(np.all(target_coord == riv_nxlonlat_cropped, axis=2))
                if len(matched_coords) == 0:
                    break
                unvisited_matched = [tuple(coord) for coord in matched_coords if tuple(coord) not in visited_coords]
                if not unvisited_matched:
                    break
                rivara_values = [g_rivara_cropped[coord[0], coord[1]] for coord in unvisited_matched]
                max_index = np.argmax(rivara_values)
                best_coord = unvisited_matched[max_index]
                riv_path_array_B[best_coord[0], best_coord[1]] = uid
                target_coord = best_coord 
                path_coords.append(target_coord)

            path_dict[uid] = path_coords

#---------------------------------------------------------------------------------------------------------------
#   Explore josui grids (josui_lst)
#---------------------------------------------------------------------------------------------------------------

    # determine josui place
    josui_lst = []

    # loop uid
    for key_num in unique_values_B:
        # get river path
        indices = np.argwhere(riv_path_city_B == key_num)
        # get minmum river area
        rivara_values = [g_rivara_cropped[coord[0], coord[1]] for coord in indices]
        min_arg = np.argmin(rivara_values)
        josui = indices[min_arg]
        # add to list
        josui_lst.append(josui)

#---------------------------------------------------------------------------------------------------------------
#   Josui map 24 x 24 (josui_array)
#---------------------------------------------------------------------------------------------------------------

    josui_array = np.ma.masked_all(rivara_max_array_B.shape, dtype='float32')

    for matching_position, uid in zip(josui_lst, unique_values_B):
        josui_array[matching_position[0], matching_position[1]] = uid

#---------------------------------------------------------------------------------------------------------------
#   Save file (josui_array)
#---------------------------------------------------------------------------------------------------------------

    josui_for_save = np.ma.masked_all(g_rivara.shape, dtype='float32')

    josui_for_save[lat_start:lat_end, lon_start:lon_end] = np.flipud(josui_array)

    josui_for_save = np.ma.filled(josui_for_save, fill_value=0)
    josui_for_save = np.where(josui_for_save > 0, 1, josui_for_save)


    josui_for_save = np.flipud(josui_for_save)

    # city purification plant
    save_path = f'/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/dat/cty_prf_/{POP}/city_{city_num:08d}.gl5'

    # save_flag
    if save_flag is True:
        josui_for_save.astype(np.float32).tofile(save_path)
        print(f"{save_path} saved")
    else:
        print('josui save_flag is false')

#---------------------------------------------------------------------------------------------------------------
#   Save file (gesui_array=rivara_max_array_B)
#---------------------------------------------------------------------------------------------------------------

    gesui_for_save = np.ma.masked_all(g_rivara.shape, dtype='float32')

    gesui_for_save[lat_start:lat_end, lon_start:lon_end] = np.flipud(rivara_max_array_B)

    gesui_for_save = np.ma.filled(gesui_for_save, fill_value=0)
    gesui_for_save = np.where(gesui_for_save > 0, 1, gesui_for_save)

    gesui_for_save = np.flipud(gesui_for_save)

    # city purification plant
    save_path = f'/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/dat/cty_swg_/{POP}/city_{city_num:08d}.gl5'

    # save_flag
    if save_flag is True:
        gesui_for_save.astype(np.float32).tofile(save_path)
        print(f"{save_path} saved")
    else:
        print('gesui save_flag is false')

#---------------------------------------------------------------------------------------------------------------
# Main loop
#---------------------------------------------------------------------------------------------------------------

def main():
#---------------------------------------------------------------------------------------------------------------
#   Initialization
#---------------------------------------------------------------------------------------------------------------

    save_flag = True
    remove_grid = 5 # minimum number of grids in one basin
    innercity_grid = 3 # minimum number of main river grid within city mask
    width = 1 # lonlat delta degree from city center

#---------------------------------------------------------------------------------------------------------------
#   loop start
#---------------------------------------------------------------------------------------------------------------

    # number of the city (1-1860)
    for target_index in range(1, 1861):
        explore(target_index, remove_grid, innercity_grid, width, save_flag=save_flag)


if __name__ == '__main__':
    main()

"""
Author Kajiyama @ 20240402
edited by Kajiyama @ 20240430
edited by Kajiyama @ 20241120
+ if grid_num = 0, no PRF
+ if grid_num = 1, PRF=SWG
+ if grid_num = 2  PRF=higher elevation, SWG=lower elevation
+ No main river, PRF=highest elevation in same basin, SWG=lowest elevation in same basin
+ No main river & no same basin, PRF=highest elevation in MASK, SWG=lowest elevation MASK
"""

import os
import copy
import numpy as np

#---------------------------------------------------------------------------------------------------------------
#   Main Function
#---------------------------------------------------------------------------------------------------------------

def explore(target_index, remove_grid, innercity_grid, width, save_flag=False):
    """
    A: After over remove_grid process
    B: After over remove_grid process & over innercity_grid process
    """
    latgrd = 2160 # sum of latitude grids (y)
    longrd = 4320 # sum of longitude grids (x)

#---------------------------------------------------------------------------------------------------------------
#   PATH
#---------------------------------------------------------------------------------------------------------------

    # root directory
    root_dir         = "/mnt/c/Users/tsimk/Downloads/dotfiles/h08/camacity"
    glob_dir         = "/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city"
    # lonlat data
    city_path        = f"{root_dir}/dat/cty_lst_/create_text/camacity_third.txt"
    # city mask data
    cmsk_dir         = f"{root_dir}/dat/cty_msk_"
    # elevation data
    elvmin_path      = f"{glob_dir}/dat/elv_min_/elevtn.CAMA.gl5"
    # riv data
    rivnum_path      = f"{glob_dir}/dat/riv_num_/rivnum.CAMA.gl5"
    rivara_path      = f"{glob_dir}/dat/riv_ara_/rivara.CAMA.gl5"
    rivnxl_path      = f"{glob_dir}/dat/riv_nxl_/rivnxl.CAMA.gl5"
    # facility data
    prf_save_dir     = f"{root_dir}/dat/cty_prf_"
    swg_save_dir     = f"{root_dir}/dat/cty_swg_"
    # auxiliary data
    nonprf_path      = f"{root_dir}/dat/non_prf_/nonprf_flag.txt"

#---------------------------------------------------------------------------------------------------------------
#   City Lon Lat Information
#---------------------------------------------------------------------------------------------------------------

    """
    first line of lines
    1|VALD|DOWN|36859.626|36855016.0|31821846.0|Japan|93.0|eastern asia| 35.6895|139.6917|Tokyo|East Asia
    """

    # city_list.txtを開いてデータを読み取る
    with open(city_path, "r") as input_file:
        lines = input_file.readlines()

    line       = lines[target_index-1] # 対象となる都市の情報行を参照
    parts      = line.split('|') # 各行をスペースで分割
    parts      = [item.strip() for item in parts]
    city_num   = int(parts[0]) # 都市番号
    cnt_lat    = float(parts[9]) # 都市中心の緯度
    cnt_lon    = float(parts[10]) # 都市中心の経度
    city_name  = parts[11].replace("\"", "").replace("?", "").replace("/", "") # 都市名
    ovlp_state = parts[1]
    clst_state = parts[2]

    # widthを使用して外枠の座標を計算
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
    # 緯度経度の始点グリッドのインデックス
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

    # 計算領域である正方形の一辺に含まれるグリッド数 (1degree = 12 grids x 12 grids)
    width_grid = width * 12 * 2

    # 緯度経度の終点グリッドのインデックス
    lat_end = lat_start + width_grid
    lon_end = lon_start + width_grid

#---------------------------------------------------------------------------------------------------------------
#   Load city mask data (g_mask_cropped)
#---------------------------------------------------------------------------------------------------------------

    if clst_state == 'NoMK' or ovlp_state == 'RMVD':
        if city_num == 1:
            with open(nonprf_path, 'w') as file:
                file.write(f"{city_num}|NoMASK\n")
        else:
            with open(nonprf_path, 'a') as file:
                file.write(f"{city_num}|NoMASK\n")
        josui_for_save = np.zeros((2160, 4320))
        gesui_for_save = np.zeros((2160, 4320))
        return josui_for_save, gesui_for_save

    g_mask_original = np.fromfile(f'{cmsk_dir}/city_clrd0000.gl5', 'float32').reshape(latgrd, longrd)
    g_mask = np.flipud(g_mask_original)
    g_mask = np.ma.masked_where(g_mask != city_num, g_mask)
    g_mask_cropped = g_mask[lat_start:lat_end, lon_start:lon_end]
    g_mask_cropped = np.flipud(g_mask_cropped)

#---------------------------------------------------------------------------------------------------------------
#   external operation
#---------------------------------------------------------------------------------------------------------------

    grid_count = int(float(parts[7]))
    if grid_count == 1:
        if city_num == 1:
            with open(nonprf_path, 'w') as file:
                file.write(f"{city_num}|True\n")
        else:
            with open(nonprf_path, 'a') as file:
                file.write(f"{city_num}|True\n")

        josui_for_save = np.zeros((2160, 4320))
        gesui_for_save = np.zeros((2160, 4320))

        coordinates = np.where(g_mask_original==city_num)
        josui_for_save[coordinates] = city_num
        gesui_for_save[coordinates] = city_num

        return josui_for_save, gesui_for_save

#---------------------------------------------------------------------------------------------------------------
#   Load elevation data (g_elv_cropped)
#---------------------------------------------------------------------------------------------------------------

    g_elv = np.fromfile(elvmin_path, 'float32').reshape(latgrd, longrd)
    g_elv = np.flipud(g_elv)
    g_elv = np.ma.masked_where(g_elv >= 1E20, g_elv)
    g_elv_cropped = g_elv[lat_start:lat_end, lon_start:lon_end]
    g_elv_cropped = np.flipud(g_elv_cropped)

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

    g_rivnum_cropped_city = np.where(g_mask_cropped == city_num, g_rivnum_cropped, np.nan)
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
#  　Basin over remove_grid (Rivnum_A_array)
#---------------------------------------------------------------------------------------------------------------

    # g_ara_num_croppedのrivnumをマスク付き配列として取得
    g_rivnum_cropped_masked = np.ma.masked_array(g_rivnum_cropped, np.isnan(g_rivnum_cropped))

    # マスクされていない要素(Nanじゃない値)のユニークな値とその出現回数を取得
    unique_values_org, counts_org = np.unique(g_rivnum_cropped_masked.compressed(), return_counts=True)
    value_counts_dict = dict(zip(unique_values_org, counts_org))

    # 値（個数）の多い順にソート
    # 都市マスク内の流域番号で，出現回数が多い順に並んでいるはず
    sorted_dict_by_value_descending = dict(sorted(value_counts_dict.items(), key=lambda item: item[1], reverse=True))

    # 値（個数）がremove grid以上の項目のみを持つ新しい辞書を作成
    # 流域が小さい物は削除する作業に該当
    filtered_dict_A = {key: value for key, value in sorted_dict_by_value_descending.items() if value >= remove_grid}

    # 空っぽのマスク配列(24x24を作る)
    Rivnum_A_array = np.ma.masked_all(g_rivnum_cropped_masked.shape, dtype='float32')

    # filtered_dict_Aのキー(流域ID)に対して繰り返し処理を行い、
    # それぞれのrivnumがg_rivnum_cropped_maskedに存在する位置を特定します。
    for rivnum_id in filtered_dict_A.keys():
        # 同じrivnumの位置を取得
        matching_positions = np.where(g_rivnum_cropped_masked.data == rivnum_id)
        # これらの位置に新しい配列にrivnumを設定
        Rivnum_A_array[matching_positions] = rivnum_id

    # Rivnum_A_arrayは都市マスクなしのすべての流域
    Rivnum_A_array = np.ma.masked_where(~np.isfinite(Rivnum_A_array) | (Rivnum_A_array == 0), Rivnum_A_array)

#---------------------------------------------------------------------------------------------------------------
#   Basin over remove_grid within city mask (Rivnum_A_array_citymasked)
#---------------------------------------------------------------------------------------------------------------

    # Rivnum_A_arrayの値が存在しないか、値が0の場所をTrueとするマスクを作成
    invalid_mask = np.isnan(Rivnum_A_array) | (Rivnum_A_array == 0)
    # g_mask_croppedが1でない場所、または上記のマスクがTrueの場所をマスクとして指定
    Rivnum_A_array_citymasked = np.ma.masked_where((g_mask_cropped != city_num) | invalid_mask, Rivnum_A_array)

#---------------------------------------------------------------------------------------------------------------
#   マスクされていない要素のユニークな値とその出現回数を取得(unique_values_A)
#---------------------------------------------------------------------------------------------------------------

    unique_values_A, counts_A = np.unique(Rivnum_A_array_citymasked.compressed(), return_counts=True)
    value_counts_dict_A = dict(zip(unique_values_A, counts_A))

#---------------------------------------------------------------------------------------------------------------
#   rivaraを使って河口グリッドを探索する (rivara_max_array_A)
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
#   各流域の経路座標　(path_dict)
#   各経路が流域番号で格納され，1つのファイルに集約 (riv_path_array_A)
#---------------------------------------------------------------------------------------------------------------

    # 保存用の変数を設定
    path_dict = {}
    riv_path_array_A = np.ma.masked_all(rivara_max_array_A.shape, dtype='float32')
    visited_coords = set()

    # マスク内の流域IDごとにループ
    for uid in unique_values_A:
        # 河口グリッドのインデックス
        coords_a = np.argwhere(rivara_max_array_A == uid)
        riv_path_array_A[coords_a[0][0], coords_a[0][1]] = uid
        if coords_a.size > 0:
            target_coord = tuple(coords_a[0]) 
            path_coords = [target_coord]
            for _ in range(len(g_mask_cropped)):
                if target_coord in visited_coords:
                    break
                visited_coords.add(target_coord)
                # riv_nxlonlat_croppedはrivnxlのlonlat表示なので
                # target_coordを次のセルに指し示すrivnxlのインデックスを取得
                matched_coords = np.argwhere(np.all(target_coord == riv_nxlonlat_cropped, axis=2))
                if len(matched_coords) == 0:
                    break
                # マッチしたインデックスの中でrivaraが最大のものを選ぶ
                unvisited_matched = [tuple(coord) for coord in matched_coords if tuple(coord) not in visited_coords]
                if not unvisited_matched:
                    break
                # g_rivara_croppedに座標をいれて，最大最小を比べている
                rivara_values = [g_rivara_cropped[coord[0], coord[1]] for coord in unvisited_matched]
                max_index = np.argmax(rivara_values)
                best_coord = unvisited_matched[max_index]
                # 河口グリッドのファイルに経路をそれぞれ足していく
                riv_path_array_A[best_coord[0], best_coord[1]] = uid
                target_coord = best_coord 
                # path_coordに経路を足していく
                path_coords.append(target_coord)

            # 各流域の経路を保存
            path_dict[uid] = path_coords

#---------------------------------------------------------------------------------------------------------------
#   Rivpath over innercity_grid (riv_path_city_B)
#---------------------------------------------------------------------------------------------------------------

    # city mask
    fill_value = 1e20
    riv_path_array_filled = riv_path_array_A.filled(fill_value)
    riv_path_city_A = np.where(g_mask_cropped==city_num, riv_path_array_filled, fill_value)

    # make new array
    riv_path_city_B = copy.deepcopy(riv_path_city_A)

    for uid in unique_values_A:
        count = 0
        mask = (riv_path_city_A == uid)
        count = np.sum(mask)
        # もし主河道のセル数が都市マスク内で指定の値より少ない場合削除
        if count < innercity_grid:
            riv_path_city_B[riv_path_city_B== uid] = fill_value

    riv_path_city_B = np.ma.masked_where(riv_path_city_B >= fill_value, riv_path_city_B)

#---------------------------------------------------------------------------------------------------------------
#   Update unique river basin number after 2 removing process (unique_values_B)
#---------------------------------------------------------------------------------------------------------------

    # compressed()を行わないとマスク値がunique_valueとしてカウントされてしまう
    unique_values_B, _ = np.unique(riv_path_city_B.compressed(), return_counts=True)

#---------------------------------------------------------------------------------------------------------------
#   都市マスク内に存在する流域を全範囲で取得(Rivnum_B_array)
#---------------------------------------------------------------------------------------------------------------

    # 新しい配列を作成
    Rivnum_B_array = np.ma.masked_all(g_rivnum_cropped_masked.shape, dtype='float32')

    # Rivnum_A_arrayに存在する新しいunique_id地点のみを保存
    for uid in unique_values_B:
        row_indices, col_indices = np.where(Rivnum_A_array == uid)
        Rivnum_B_array[row_indices, col_indices] = uid

#---------------------------------------------------------------------------------------------------------------
#   Updated river mouse grid (rivara_max_array_B)
#---------------------------------------------------------------------------------------------------------------

    # データ方とサイズに基づいて新しい配列を作成
    rivara_max_array_B = np.ma.masked_all(g_ara_num_cropped.shape, dtype='float32')

    for rivnum_id in unique_values_B:
        # 同じrivnumの位置を取得
        matching_positions = np.where(Rivnum_A_array_citymasked == rivnum_id)
        # これらの位置におけるrivaraの最大値の位置を取得
        max_rivara_position = np.argmax(g_rivara_cropped[matching_positions])
        # 最大のrivaraの位置に対応するrivnumを新しい配列に保存する
        rivara_max_array_B[matching_positions[0][max_rivara_position], matching_positions[1][max_rivara_position]] = rivnum_id

#---------------------------------------------------------------------------------------------------------------
#   Update riv_path_array with full length out of city mask (riv_path_array_B)
#---------------------------------------------------------------------------------------------------------------

    # 保存用の変数を設定
    path_dict = {}
    riv_path_array_B = np.ma.masked_all(rivara_max_array_B.shape, dtype='float32')
    visited_coords = set()

    # マスク内の流域IDごとにループ
    for uid in unique_values_B:
        # 河口グリッドのインデックス
        coords_a = np.argwhere(rivara_max_array_B == uid)
        riv_path_array_B[coords_a[0][0], coords_a[0][1]] = uid
        if coords_a.size > 0:
            target_coord = tuple(coords_a[0]) 
            path_coords = [target_coord]
            for _ in range(len(g_mask_cropped)):
                if target_coord in visited_coords:
                    break
                visited_coords.add(target_coord)
                # riv_nxlonlat_croppedはrivnxlのlonlat表示なので
                # target_coordを次のセルに指し示すrivnxlのインデックスを取得
                matched_coords = np.argwhere(np.all(target_coord == riv_nxlonlat_cropped, axis=2))
                if len(matched_coords) == 0:
                    break
                # マッチしたインデックスの中でrivaraが最大のものを選ぶ
                unvisited_matched = [tuple(coord) for coord in matched_coords if tuple(coord) not in visited_coords]
                if not unvisited_matched:
                    break
                # g_rivara_croppedに座標をいれて，最大最小を比べている
                rivara_values = [g_rivara_cropped[coord[0], coord[1]] for coord in unvisited_matched]
                max_index = np.argmax(rivara_values)
                best_coord = unvisited_matched[max_index]
                # 河口グリッドのファイルに経路をそれぞれ足していく
                riv_path_array_B[best_coord[0], best_coord[1]] = uid
                target_coord = best_coord 
                # path_coordに経路を足していく
                path_coords.append(target_coord)

            # 各流域の経路を保存
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

    # 浄水場情報を24x24のマスクファイルに保存
    josui_array = np.ma.masked_all(rivara_max_array_B.shape, dtype='float32')

    for matching_position, uid in zip(josui_lst, unique_values_B):
        josui_array[matching_position[0], matching_position[1]] = uid

#---------------------------------------------------------------------------------------------------------------
#   Gesui map 24 x 24 (josui_array)
#---------------------------------------------------------------------------------------------------------------

    gesui_array = copy.deepcopy(rivara_max_array_B)

#---------------------------------------------------------------------------------------------------------------
#   Check whehter no prf
#---------------------------------------------------------------------------------------------------------------

    # remote aqueductの時に同流域を探索するかどうかのflag
    no_prf_flag = False

    # josui_arrayに値が存在するかどうか
    josui_array = np.ma.filled(josui_array, fill_value=0)
    prf_coords = np.where(josui_array>0)

    if len(prf_coords[0]) == 0:
        print(f"no purification")

        # intake_exploreでの同流域探索をon
        no_prf_flag = True

        josui_array, gesui_array = explore_prf(g_mask_cropped, 
                                               city_num,
                                               g_rivnum_cropped_city, 
                                               g_elv_cropped, 
                                               g_rivara_cropped,
                                               )
        print(f"non_prf -> tentative prf")

#---------------------------------------------------------------------------------------------------------------
#   Check whehter no prf
#---------------------------------------------------------------------------------------------------------------

    # text save
    if save_flag is True:
        if target_index == 1:
            with open(nonprf_path, 'w') as file:
                file.write(f"{target_index}|{no_prf_flag}\n")
        else:
            with open(nonprf_path, 'a') as file:
                file.write(f"{target_index}|{no_prf_flag}\n")
    else:
        print('nonprf save_flag is false')

#---------------------------------------------------------------------------------------------------------------
#   Save file (josui_array)
#---------------------------------------------------------------------------------------------------------------

    """
    croppするときは必ずひっくり返す
    保存・描写するときにもとに戻す
    """

    # 保存用ファイル作成
    josui_for_save = np.ma.masked_all(g_rivara.shape, dtype='float32')

    #　cropp区間の値を変換(世界地図はひっくり返っている)
    josui_for_save[lat_start:lat_end, lon_start:lon_end] = np.flipud(josui_array)

    # 浄水場を1, それ以外を0とするバイナリーファイルに変換
    josui_for_save = np.ma.filled(josui_for_save, fill_value=0)
    josui_for_save = np.where(josui_for_save > 0, city_num, josui_for_save)

    # 保存するときは世界地図をひっくり返して，正しい向きにしておく
    josui_for_save = np.flipud(josui_for_save)


#---------------------------------------------------------------------------------------------------------------
#   Save file (gesui_array=rivara_max_array_B)
#---------------------------------------------------------------------------------------------------------------

    """
    croppするときは必ずひっくり返す
    保存・描写するときにもとに戻す
    """

    # 保存用ファイル作成
    gesui_for_save = np.ma.masked_all(g_rivara.shape, dtype='float32')

    #　cropp区間の値を変換(世界地図はひっくり返っている)
    gesui_for_save[lat_start:lat_end, lon_start:lon_end] = np.flipud(gesui_array)

    # 浄水場を1, それ以外を0とするバイナリーファイルに変換
    gesui_for_save = np.ma.filled(gesui_for_save, fill_value=0)
    gesui_for_save = np.where(gesui_for_save > 0, city_num, gesui_for_save)

    # 保存するときは世界地図をひっくり返して，正しい向きにしておく
    gesui_for_save = np.flipud(gesui_for_save)


    return josui_for_save, gesui_for_save

#---------------------------------------------------------------------------------------------------------------
# MODULES
#---------------------------------------------------------------------------------------------------------------

def l_coordinate_to_tuple(lcoordinate, a=2160, b=4320):
    lat_l = a - ((lcoordinate - 1) // b)
    lon_l = (lcoordinate) % b - 1
    return (lat_l, lon_l)


def explore_prf(citymask, city_num, rivnum, elevation, rivara):
    """
    citymask:  g_mask_cropped,             city_mask
    rivnum:    g_rivnum_cropped_city,      city_mask内のrivnumデータ
    elevation: g_elv_cropped,              elevationデータ
    rivara:    g_rivara_cropped,           rivaraデータ
    """

    # rivnum_cityの流域番号をkey, 各流域のグリッド数をvalueに持つdictionary
    unique_values, counts = np.unique(rivnum.compressed(), return_counts=True)
    uid_dict = dict(zip(unique_values, counts))

    # 流域グリッドが最大のkeyを見つける
    max_key = max(uid_dict, key=uid_dict.get)

    # 流域が2グリッド以上存在することを確認
    if max_key > 1:

        # 選ばれた流域内のelevation
        elv_indices = np.argwhere(rivnum == max_key)
        elv_values = [elevation[coord[0], coord[1]] for coord in elv_indices]

        # 標高最大の点　josui
        elv_maxarg = np.argmax(elv_values)
        josui_coord = elv_indices[elv_maxarg]
        josui_array = np.zeros(rivnum.shape, dtype='float32')
        josui_array[josui_coord[0], josui_coord[1]] = max_key

        # 標高最大以外で集水面積が一番大きい場所(河口)
        ara_indices = np.argwhere((citymask == city_num) & (josui_array != rivnum[josui_coord[0], josui_coord[1]]))
        ara_values = [rivara[coord[0], coord[1]] for coord in ara_indices]
        # 空じゃないか確かめる
        if ara_values:
            ara_argmax = np.argmax(ara_values)
            gesui_coord = ara_indices[ara_argmax]
        else:
            # 標高最小を選ぶ
            print(f"ara_indices is empty -> argmin_elv for gesui")
            elv_minarg = np.argmax(elv_values)
            gesui_coord = elv_indices[elv_minarg]

    # すべての流域が1グリッド以下であるとき
    else:

        # city mask内のelevatoin
        elv_indices = np.argwhere(citymask == city_num)
        elv_values = [elevation[coord[0], coord[1]] for coord in elv_indices]

        # josui
        elv_maxarg = np.argmax(elv_values)
        josui_coord = elv_indices[elv_maxarg]
        josui_array = np.zeros(rivnum.shape, dtype='float32')
        josui_array[josui_coord[0], josui_coord[1]] = rivnum[josui_coord[0], josui_coord[1]]

        # 標高最大以外で集水面積が一番大きい場所(河口)
        ara_indices = np.argwhere((citymask == city_num) & (josui_array != rivnum[josui_coord[0], josui_coord[1]]))
        ara_values = [rivara[coord[0], coord[1]] for coord in ara_indices]
        # 空じゃないか確かめる
        if ara_values:
            ara_argmax = np.argmax(ara_values)
            gesui_coord = ara_indices[ara_argmax]
        else:
            # 標高最小を選ぶ
            print(f"ara_indices is empty -> argmin_elv for gesui")
            elv_minarg = np.argmax(elv_values)
            gesui_coord = elv_indices[elv_minarg]

    # gesui
    gesui_array = np.ma.masked_all(rivnum.shape, dtype='float32')
    gesui_array[gesui_coord[0], gesui_coord[1]] = rivnum[gesui_coord[0], gesui_coord[1]]

    # josui
    josui_array = np.ma.masked_all(rivnum.shape, dtype='float32')
    josui_array[josui_coord[0], josui_coord[1]] = rivnum[josui_coord[0], josui_coord[1]]

    return josui_array, gesui_array

#---------------------------------------------------------------------------------------------------------------
# Main loop
#---------------------------------------------------------------------------------------------------------------

def main():

#---------------------------------------------------------------------------------------------------------------
#   Initialization
#---------------------------------------------------------------------------------------------------------------

    save_flag = True
    remove_grid = 5 # minimum number of grids in one basin
    innercity_grid = 2 # minimum number of main river grid within city mask
    width = 2 # lonlat delta degree from city center

#---------------------------------------------------------------------------------------------------------------
#   save file
#---------------------------------------------------------------------------------------------------------------
    canvas_prf = np.zeros((2160, 4320))
    canvas_swg = np.zeros((2160, 4320))

#---------------------------------------------------------------------------------------------------------------
#   loop start
#---------------------------------------------------------------------------------------------------------------

    # number of the city (1-1860)
    for target_index in range(1, 1861):
        josui_for_save, gesui_for_save = explore(target_index, remove_grid, innercity_grid, width, save_flag=save_flag)

        non_zero_prf = np.where(josui_for_save != 0)
        non_zero_swg = np.where(gesui_for_save != 0)
        canvas_prf[non_zero_prf] = target_index
        canvas_swg[non_zero_swg] = target_index

#---------------------------------------------------------------------------------------------------------------
#   save file
#---------------------------------------------------------------------------------------------------------------

    root_dir         = "/mnt/c/Users/tsimk/Downloads/dotfiles/h08/camacity"
    prf_save_dir     = f"{root_dir}/dat/cty_prf_"
    swg_save_dir     = f"{root_dir}/dat/cty_swg_"

    prf_save_path = f'{prf_save_dir}/prf_clrd0000.gl5'
    swg_save_path = f'{swg_save_dir}/swg_clrd0000.gl5'

    if save_flag is True:
        canvas_prf.astype(np.float32).tofile(prf_save_path)
        canvas_swg.astype(np.float32).tofile(swg_save_path)
        print(f"{prf_save_path}, {swg_save_path} saved")
    else:
        print('save_flag is false')

if __name__ == '__main__':
    main()

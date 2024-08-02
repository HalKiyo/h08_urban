######################################################################
# Before run this script, please prepare (pre.py)basin_to_cities data
# This script summarize wsi/pre/updown/abandon_prf_or_intake.ipynb
######################################################################

import os
import json
import numpy as np
import pandas as pd

def main():
    #
    distance = 100
    # init
    h08dir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08'
    rivnxl_path = f"{h08dir}/global_city/dat/riv_nxl_/rivnxl.CAMA.gl5"
    rivout_path = f'{h08dir}/wsi/dat/riv_out_/W5E5LR__00000000.gl5'
    rivnum_path = f'{h08dir}/global_city/dat/riv_num_/rivnum.CAMA.gl5'
    rivara_path = f'{h08dir}/global_city/dat/riv_ara_/rivara.CAMA.gl5'
    prf_dir = f'{h08dir}/global_city/dat/cty_prf_'
    int_dir = f'{h08dir}/global_city/dat/cty_int_/{distance}km_samebasin'

    rivout_gl5 = np.fromfile(rivout_path, dtype='float32').reshape(2160, 4320)
    rivnum_gl5 = np.fromfile(rivnum_path, dtype='float32').reshape(2160, 4320)
    rivara_gl5 = np.fromfile(rivara_path, dtype='float32').reshape(2160, 4320)
    rivnxl_gl5 = np.fromfile(rivnxl_path, 'float32').reshape(2160, 4320)
    riv_nxlonlat_cropped = nxtl2nxtxy(rivnxl_gl5, 0, 0)

    mcy_pop_water_path = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/dat/cty_lst_/gpw4/mcy_pop_water_12region.txt'
    column_names = ['index', 'flag', 'rate', 'wup', 'pop', 'ava', 'mpcy', 'potential', 'region', 'country', 'cityname', '12region']
    df = pd.read_csv(mcy_pop_water_path, delimiter='|', header=None, names=column_names)


    ####################################################################################
    # JOB
    ####################################################################################

    # 保存用変数を作る
    df_new = df.copy()

    # 複数都市を有する流域データを読み込む
    new_basin_to_cities = load_json() # keys = all basins with multiple cities 180?

    # basin loop
    for key_index in range(len(new_basin_to_cities.keys())):

        # result_dictが更新されないことが大事
        result_dict =  make_result_dict(new_basin_to_cities, rivout_gl5, riv_nxlonlat_cropped, key_index=key_index) # keys = all cities in key_index basin

        # rivnumを取得する
        riv_num = list(new_basin_to_cities.keys())[key_index]
        city_num_lst = list(result_dict.keys())

        # city_numを取得する
        for city_num in city_num_lst:

            # get default discharge
            rate, discharge = discharge_rate(city_num, riv_num, rivout_gl5, rivnum_gl5, rivara_gl5)

            # multiple basin cities
            print(f'city_num: {city_num}, riv_num: {riv_num}')
            print(f'rate {rate}')
            print(f'discharge {discharge/1e9}')

            # check if df works OK
            filtered_df = df[df['index'] == city_num]
            if filtered_df.empty:
                print(f'City number {city_num} not found in the data.')

            # population in downtown
            est_pop = filtered_df.reset_index(drop=True).loc[0, 'pop']

            # calc demand from one basin(riv_num)
            demand = est_pop*rate*1000

            # ロジックを考える
            #demandが1000m3/year * populationだとして
            #consumptionはsupply > demand or supply < demandによって変化するべき
            #consumptionについてはexplore内で計算する必要がある

            print(f'supply: {discharge/1e9}')
            print(f'demand: {demand/1e9}')
            result_dict[city_num][0] = discharge
            result_dict[city_num][1] = demand
            print(f'--------------------------------------')

        print(f'**********rivnum{list(new_basin_to_cities.keys())[key_index]}: discharge done***************')
        print(f'--------------------------------------')

        # explore
        result_dict_mod = explore(result_dict)

        # update availability of cities in result_dict_mod
        for city_num in result_dict_mod:
            city_num = int(city_num)
            riv_num = int(float(riv_num))

            h08dir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08'

            prf_dir = f'{h08dir}/global_city/dat/cty_prf_'
            prf_path = f'{prf_dir}/vld_cty_/city_{city_num:08}.gl5'
            prf = np.fromfile(prf_path, dtype='float32').reshape(2160, 4320)

            int_dir = f'{h08dir}/global_city/dat/cty_int_/{distance}km_samebasin'
            int_path = f'{int_dir}/city_{city_num:08}.gl5'
            intake = np.fromfile(int_path, dtype='float32').reshape(2160, 4320)

            flag, largest, all_int = prf_int_flag(prf, intake, riv_num, rivnum_gl5, rivout_gl5)

            # discharge of prfs and intake
            els_prf = np.sum(rivout_gl5[(prf==1) & (rivnum_gl5!=riv_num)])* 60 * 60 * 24 * 365 / (1000) # m3/year
            dis_int = np.sum(rivout_gl5[(intake==1) & (rivnum_gl5!=riv_num)])* 60 * 60 * 24 * 365 / (1000) # m3/year

            # 同一流域にintakeとprfがあり: prfを選択
            if flag == 'both-prf':
                dis_els = els_prf
                ibt = 0
            if flag == 'both-intake':
                dis_els = els_prf
                ibt = 0
            if flag == 'prf':
                dis_els = els_prf
                ibt = dis_int
            if flag == 'intake':
                dis_els = els_prf
                ibt = 0

            # 都市がriv_num以外にprfを持つ場合
            dis_mod = result_dict_mod[city_num][0]
            if dis_els:
                availability = dis_els + dis_mod
            else:
                availability = dis_mod

            # update
            new_mpcy = availability/df_new.loc[df_new['index'] == city_num, 'pop']
            new_mpcy = new_mpcy.iloc[0]
            ava_int = availability + ibt
            mpcy_int = ava_int/df_new.loc[df_new['index'] == city_num, 'pop']
            mpcy_int = mpcy_int.iloc[0]

            print(f'city_num: {city_num}')
            print(f'dis_els:{dis_els/1e9} billion m3/year')
            print(f'dis_mod:{dis_mod/1e9} billion m3/year')
            print(f'ibt: {ibt/1e9} billion m3/year')
            print(f'availability:{availability/1e9} billion m3/year')
            print(f"new_mpcy: {new_mpcy}")
            print(f'ava_int: {ava_int/1e9} billion m3/year')
            print(f"mpcy_int: {mpcy_int}")
            print(f'--------------------------------------')

            # overwrite df_new
            df_new.loc[df_new['index'] == city_num, 'ava'] = availability
            df_new.loc[df_new['index'] == city_num, 'mpcy'] = new_mpcy
            df_new.loc[df_new['index'] == city_num, 'ava_int'] = ava_int
            df_new.loc[df_new['index'] == city_num, 'mpcy_int'] = mpcy_int

        print(f'**********rivnum{list(new_basin_to_cities.keys())[key_index]}: update done***************')

    return df_new

def l_coordinate_to_tuple(lcoordinate, a=2160, b=4320):
    lat_l = ((lcoordinate - 1) // b)
    lon_l = (lcoordinate) % b - 1
    return (lat_l, lon_l)

def nxtl2nxtxy(rgnfile, upperindex, leftindex):
    vfunc = np.vectorize(l_coordinate_to_tuple, otypes=[tuple])
    riv_nxtxy = np.empty(rgnfile.shape, dtype=tuple)
    mask = ~np.isnan(rgnfile)
    riv_nxtxy[mask] = vfunc(rgnfile[mask])
    riv_nxtxy_shape = (riv_nxtxy.shape[0], riv_nxtxy.shape[1], 2)

    riv_nxtxy_lst = []
    for row in riv_nxtxy:
        for y, x in row:
            modified_y = y - upperindex
            modified_x = x - leftindex
            riv_nxtxy_lst.append((modified_y, modified_x))

    riv_nxtxy_cropped = np.array(riv_nxtxy_lst).reshape(riv_nxtxy_shape)
    riv_nxtxy_cropped = riv_nxtxy_cropped.astype(int)
    return riv_nxtxy_cropped

def load_json(distance=100):
    h08dir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08'
    basin_cities_path = f'{h08dir}/global_city/dat/riv_num_/basin_to_cities_{distance}km.json'
    with open(basin_cities_path, 'r') as json_file:
        basin_to_cities = json.load(json_file)
    new_basin_to_cities = {rivnum: cities for rivnum, cities in basin_to_cities.items() if len(cities) > 1}
    return new_basin_to_cities

def prf_int_flag(prf, intake, riv_num, rivnum_gl5, rivout_gl5):
    prf_coord = np.where((prf == 1) & (rivnum_gl5 == riv_num))
    int_coord = np.where((intake==1) & (rivnum_gl5 == riv_num))

    if prf_coord[0].size != 0 and int_coord[0].size != 0:

        prf_runout = rivout_gl5[prf_coord][0] * 60 * 60 * 24 * 365 / (1000)
        int_runout = rivout_gl5[int_coord][0] * 60 * 60 * 24 * 365 / (1000)

        if prf_runout > int_runout:
            print(f'prf_runout > int_runout')
            largest = prf
            all_int = prf
            flag = 'both-prf'
        else:
            print(f'int_runout > prf_runout')
            prf[prf_coord] = 0
            largest = intake
            all_int = intake
            flag = 'both-intake'

    elif prf_coord[0].size != 0 and int_coord[0].size == 0:
        flag = 'prf'
        largest = prf
        all_int = prf + intake

    elif prf_coord[0].size == 0 and int_coord[0].size != 0:
        flag = 'intake'
        largest = intake
        all_int = prf + intake

    else:
        print('Error no prf no int')

    return flag, largest, all_int

def is_valid_edge(city1, city2, coords_dict):
    for coord, city_list in coords_dict.items():
        if len(city_list) > 1:
            if city1 in city_list and city2 in city_list:
                if city1 > city2:
                    continue
                else:
                    city1, city2 = None, None

    if city1 and city2:
        return True
    else:
        return False

def updown(new_basin_to_cities, key_index=0, distance=100):
    # get uid and city list
    keys_list = list(new_basin_to_cities.keys())
    uid = keys_list[key_index]
    rivnum_list = new_basin_to_cities[uid]
    rivnum_list = [int(i) for i in rivnum_list]
    print(f'uid: {uid}')

    # remove overlap
    h08dir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08'
    overlap_path = f'{h08dir}/global_city/dat/cty_lst_/gpw4/overlap_hidden_only.txt'
    with open(overlap_path, 'r') as f:
        numbers = [int(line.strip()) for line in f]
    rivnum_list_removed = [num for num in rivnum_list if num not in numbers]
    rivnum_list_removed = [int(i) for i in rivnum_list_removed]
    print(f'city_list: {rivnum_list_removed}')

    # rivout
    rivout_path = f'{h08dir}/wsi/dat/riv_out_/W5E5LR__00000000.gl5'
    rivout_gl5 = np.fromfile(rivout_path, dtype='float32').reshape(2160, 4320)

    # coord of purficication
    coords_a = []
    for city_num in rivnum_list_removed:
        # rivnum
        rivnum_path = f'{h08dir}/global_city/dat/riv_num_/rivnum.CAMA.gl5'
        rivnum_gl5 = np.fromfile(rivnum_path, dtype='float32').reshape(2160, 4320)

        # prf
        prf_dir = f'{h08dir}/global_city/dat/cty_prf_'
        prf_path = f'{prf_dir}/vld_cty_/city_{city_num:08}.gl5'
        prf = np.fromfile(prf_path, dtype='float32').reshape(2160, 4320)

        # int
        int_dir = f'{h08dir}/global_city/dat/cty_int_/{distance}km_samebasin'
        int_path = f'{int_dir}/city_{city_num:08}.gl5'
        intake = np.fromfile(int_path, dtype='float32').reshape(2160, 4320)

        # 1. prfのみが対象流域内(uid)
        # 2. intのみが対象流域内(uid)
        # 3. prfとintどちらもが対象領域内(uid)
        uid = int(float(uid))
        prf_coord = np.where((prf == 1) & (rivnum_gl5 == uid))
        int_coord = np.where((intake == 1) & (rivnum_gl5 == uid))

        if prf_coord[0].size != 0 and int_coord[0].size != 0:

            prf_runout = rivout_gl5[prf_coord][0] * 60 * 60 * 24 * 365 / (1000)
            int_runout = rivout_gl5[int_coord][0] * 60 * 60 * 24 * 365 / (1000)

            if prf_runout > int_runout:
                larger_coord = prf_coord
            else:
                larger_coord = int_coord

        elif prf_coord[0].size != 0 and int_coord[0].size == 0:
            larger_coord = prf_coord

        elif prf_coord[0].size == 0 and int_coord[0].size != 0:
            larger_coord = int_coord

        else:
            larger_coord = prf_coord
            continue

        coords_a.append([larger_coord, city_num])

    ####################################################################################
    # 同流域内で取水点がかぶっている都市を取水点ごとにリストにする
    ####################################################################################

    # 同じ座標に対して city_num のリストを作成するための辞書
    coords_dict = defaultdict(list)

    # coords_a の要素をループして座標をキーに city_num をリストに追加
    for coord, city_num in coords_a:
        coord_tuple = (tuple(coord[0]), tuple(coord[1]))
        coords_dict[coord_tuple].append(city_num)

    ####################################################################################
    # Down stream explore start
    ####################################################################################

    # down
    edges = []
    riv_path_array = np.zeros((2160, 4320))

    # city_num loop 
    for idx in range(len(coords_a)):

        cityup = coords_a[idx][1]
        #print(f'-----------------------------------------------------------')
        #print(f'cityup: {cityup}')
        #print(f'coord: {(coords_a[idx][0][0][0], coords_a[idx][0][1][0])}')

        visited_coords = set()

        # coords_a[idx][0] = (array([732]), array([3086]))
        riv_path_array[coords_a[idx][0][0][0], coords_a[idx][0][1][0]] = idx

        # coordinates of prf and intake
        coords_b = coords_a.copy()
        coords_b.pop(idx)

        if len(coords_a) > 0:
            target_coord = (coords_a[idx][0][0][0], coords_a[idx][0][1][0])

            while True:
                if target_coord in visited_coords:
                    break
                visited_coords.add(target_coord)

                next_coord = riv_nxlonlat_cropped[target_coord[0], target_coord[1]]
                if next_coord.size == 0 or next_coord.shape != (2,):
                    break

                next_coord = (next_coord[0], next_coord[1])
                riv_path_array[next_coord[0], next_coord[1]] = idx
                target_coord = next_coord

        for coord in coords_b:
            citydwn = coord[1]

            if coord[0][0].size == 0:
                continue
            else:
                standard_coord = (coord[0][0][0], coord[0][1][0])
                if standard_coord in visited_coords:
                    edge_flag = is_valid_edge(cityup, citydwn, coords_dict)
                    if edge_flag:
                        #print(f'citydwn: {citydwn}')
                        edges.append((cityup, citydwn))
                    else:
                        #print(f'invalid: {citydwn}')
                        continue
                else:
                    #print(f'No: {citydwn}')
                    continue

    # edgesは2都市間のupstreamとdownstreamの関係をすべて保存したリスト
    return edges, riv_path_array, coords_a, rivnum_list_removed

def find_upstream_cities(edges, city_number):
    cities_with_upstream = []
    cities_without_upstream = []
    upstream_cities = [edge[0] for edge in edges if edge[1] == city_number]
    uup = np.unique(upstream_cities)

    for i in uup:
        if any(edge[1] == i for edge in edges):
            # 上流にさらに都市が存在する場合
            cities_with_upstream.append(i)
        else:
            # 上流にこれ以上都市が存在しない場合
            cities_without_upstream.append(i)

    return cities_with_upstream, cities_without_upstream, uup


####################################################################################
# key_index(すべての流域を)すべてを探索する
# result_dictのavaの中に1流域からの想定取水量を入力する
# result_dictを作成する
####################################################################################
def make_result_dict(new_basin_to_cities, rivout_gl5, riv_nxlonlat_cropped, key_index=1):
    # unique_cties_basin: uniqueなすべての都市
    # uup: 各都市の上流にあるuniqueな都市
    edges, riv_path_array, coords_a, rivnum_list_removed = updown(new_basin_to_cities, rivout_gl5, riv_nxlonlat_cropped, key_index=key_index)
    unique_cities_basin = np.unique(edges)

    result_dict = {}
    for city_number in unique_cities_basin:
        cities_with_upstream, cities_without_upstream, uup = find_upstream_cities(edges, city_number)
        if cities_with_upstream:
            # [discharge, consumption, flag, uppoer cities]
            result_dict[city_number] = [1, 0.1, True, uup]
        elif not cities_with_upstream and cities_without_upstream:
            result_dict[city_number] = [1, 0.1, False, uup]
        else:
            result_dict[city_number] = [1, 0.1, 'Done', uup]

    # 同じ流域に含まれる各都市のwater availability, upstream citiesを保存している．cities_with_upstreamが存在する場合，flag=Trueとなる
    return result_dict

####################################################################################
# result_dictのavaの中に1流域からの想定取水量を入力する
# 一つの流域に二つ取水点がある可能性がある→prfかintakeの流量が多い方を選ぶ
####################################################################################
def discharge_rate(city_num, riv_num, rivout_gl5, rivnum_gl5, rivara_gl5, distance=100):
    # riv_num以外からの取水がupdownでどのような影響を受けるかを考慮できない?
    # largestはriv_numでの理想流量を計算するためのもの．(prfかintかいずれかから1つのみ)
    # all_intはcityが複数のriv_numから取水している場合のすべての取水点

    city_num = int(city_num)
    riv_num = int(float(riv_num))
    print(f'city_num, riv_num, {city_num}, {riv_num}')

    h08dir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08'

    prf_dir = f'{h08dir}/global_city/dat/cty_prf_'
    prf_path = f'{prf_dir}/vld_cty_/city_{city_num:08}.gl5'
    prf = np.fromfile(prf_path, dtype='float32').reshape(2160, 4320)

    int_dir = f'{h08dir}/global_city/dat/cty_int_/{distance}km_samebasin'
    int_path = f'{int_dir}/city_{city_num:08}.gl5'
    intake = np.fromfile(int_path, dtype='float32').reshape(2160, 4320)

    flag, largest, all_int = prf_int_flag(prf, intake, riv_num, rivnum_gl5, rivout_gl5)

    # rateを計算するための変数群
    rivnum_mask = np.ma.masked_where(all_int == 0, rivnum_gl5)
    rivara_mask = np.ma.masked_where(all_int == 0, rivara_gl5) # 複数ある

    # rateはcityが有するprfとintの中で，対象のriv_numがどれだけの割合を担っているかを示す
    rate = np.sum(rivara_gl5[rivnum_mask== riv_num])/np.sum(rivara_mask)

    # dischargeはupdownを考慮しない理想化されたdischarge
    # かつ対象riv_numでのdischargeのみをreturnする
    discharge = np.sum(rivout_gl5[(largest!=0) & (rivnum_gl5==riv_num)])* 60 * 60 * 24 * 365 / (1000) # m3/year

    return rate, discharge

####################################################################################
# 上流と下流の取水の関係をrecursiveに探索
# result_dictのavaの部分を更新してreturn
# 二つの都市で，互いが互いの都市の更新を待っている状態だと，whileが無限になる

# ロジックを考える
#demandが1000m3/year * populationだとして
#consumptionはsupply > demand or supply < demandによって変化するべき
#consumptionについてはexplore内で計算する必要がある

#result_dict[city_num][0] = discharge
#result_dict[city_num][1] = demand
####################################################################################

def explore(result_dict):

    while True:
        all_done = True  # 全てのflagが'Done'であるかをチェックするためのフラグ

        for key in result_dict:
            value = result_dict[key]
            ava = value[0]
            dem = value[1]
            flag = value[2]
            uup = value[3]

            ava_mod = ava

            if flag != 'Done':
                all_done = False  # まだ'Done'でないflagがある場合、all_doneをFalseに設定

            # no upstream cities
            if flag is False:
                flg_lst = []
                for kkk in uup:
                    oth_flg = result_dict[kkk][2]
                    flg_lst.append(oth_flg)

                for kkk in uup:
                    oth_dem = result_dict[kkk][1]
                    if oth_dem > ava_mod:
                        oth_dem = ava_mod
                    ava_mod = ava_mod - np.abs(oth_dem)
                result_dict[key][0] = ava_mod

                if dem > ava_mod:
                    result_dict[key][1] = ava_mod

                result_dict[key][2] = 'Done'

            # upstream cities exist
            elif flag is True:
                flg_lst = []
                for kkk in uup:
                    oth_flg = result_dict[kkk][2]
                    flg_lst.append(oth_flg)

                if all(item == 'Done' for item in flg_lst):
                    # 重複を削除する
                    all_upstream = []
                    for kkk in uup:
                        all_upstream.extend(result_dict[kkk][2])
                    mod_uup = [city for city in uup if city not in all_upstream]
                    for kkk in mod_uup:
                        oth_dem = result_dict[kkk][1]
                        if oth_dem > ava_mod:
                            oth_dem = ava_mod
                        ava_mod = ava_mod - np.abs(oth_dem)
                    result_dict[key][0] = ava_mod

                    if dem > ava_mod:
                        result_dict[key][1] = ava_mod

                    result_dict[key][2] = 'Done'

            else:
                flg_lst = []
                for kkk in uup:
                    oth_flg = result_dict[kkk][2]
                    flg_lst.append(oth_flg)
            print(key, flag, ava, dem, flg_lst, uup)

        print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        if all_done:
            print(f'********** explore all done ***************')
            break  # 全てのflagが'Done'であればループを終了
    return result_dict

if __name__ == '__main__':
    main()

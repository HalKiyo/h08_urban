"""
Author Doi @ 20210331
modified by kajiyama @20240224
+ inter basin water transfer by shumilova 2018
modified by kajiyama @20240402
+ downtown mask
+ not to exceed ocean mask
+ earth mover's distance
modified by kajiyama @20240430
+ no prf flag action
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt


def explore(city_num, save_flag=False):
    print('-------------------------')
    print(f"city_num {city_num}")

#----------------------------------------------------------------------------------------
#   Init
#----------------------------------------------------------------------------------------

    NAME= 'W5E5'
    MAP= '.CAMA'
    SUF = '.gl5'
    dtype= 'float32'
    POP='vld_cty_'
    year_start = 2019
    year_end = 2020
    lat_num = 2160
    lon_num = 4320
    can_exp = 2    # grid radius for canal grid modification exploring
    exp_range = 24 # grid radius of exploring square area 100km
    distance_condition = 50 #km

#----------------------------------------------------------------------------------------
#   PATH
#----------------------------------------------------------------------------------------

    root_dir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city'
    dis_dir = f"{root_dir}/dat/riv_out_"
    can_in_path = f"{root_dir}/dat/can_ext_/existing_origin{SUF}"
    can_out_path = f"{root_dir}/dat/can_ext_/existing_destination_1{SUF}"
    elv_path = f"{root_dir}/dat/elv_min_/elevtn{MAP}{SUF}"
    rivnum_path = f"{root_dir}/dat/riv_num_/rivnum{MAP}{SUF}"
    nonprf_path = f"{root_dir}/dat/non_prf_/{POP}/nonprf_flag.txt"
    prf_path = f"{root_dir}/dat/cty_prf_/{POP}/city_{city_num:08}{SUF}"
    cnt_path = f"{root_dir}/dat/cty_cnt_/gpw4/modified/city_{city_num:08}{SUF}"
    msk_path = f"{root_dir}/dat/{POP}/city_{city_num:08}{SUF}"
    savepath = f"{root_dir}/dat/cty_int_/{POP}/city_{city_num:08}{SUF}"
    displaypath = f'{root_dir}/dat/cty_int_/fig/intake_display_{POP}_{city_num:08}{SUF}'

#----------------------------------------------------------------------------------------
#   Whether valid mask or not
#----------------------------------------------------------------------------------------

    if not os.path.exists(msk_path):
        print(f"{city_num} is invalid mask")
        return

#----------------------------------------------------------------------------------------
#   Load
#----------------------------------------------------------------------------------------

    # river discharge data
    for year in range(year_start, year_end, 1):

        dis_path = f"{dis_dir}/{NAME}LR__{year}0000{SUF}"
        riv_dis_tmp = np.fromfile(dis_path, dtype=dtype).reshape(lat_num, lon_num)

        if year == year_start:
            riv_dis = riv_dis_tmp
        else:
            riv_dis = riv_dis + riv_dis_tmp

    # annual average discharge
    riv_dis = riv_dis/(year_end - year_start)

    # canal map
    can_in = np.fromfile(can_in_path, dtype=dtype).reshape(lat_num, lon_num)
    can_out = np.fromfile(can_out_path, dtype=dtype).reshape(lat_num, lon_num)

    # elevation map
    elv = np.fromfile(elv_path, dtype=dtype).reshape(lat_num, lon_num)

    # water shed number map
    rivnum = np.fromfile(rivnum_path, dtype=dtype).reshape(lat_num, lon_num)

    # city mask data
    city_mask = np.fromfile(msk_path, dtype=dtype).reshape(lat_num, lon_num)

    # purification plant location
    prf = np.fromfile(prf_path, dtype=dtype).reshape(lat_num, lon_num)

    # no purification flag
    with open(nonprf_path, 'r') as input_file:
        lines = input_file.readlines()
    line = lines[city_num-1]
    parts = line.split('|')
    parts = [item.strip() for item in parts]
    no_prf_flag = parts[1]
    if no_prf_flag == 'True':
        print(f"no_prf_flag: {no_prf_flag}")

#-------------------------------------------------------------------------------------------
#   JOB
#-------------------------------------------------------------------------------------------

    # prf location
    indices = np.where(prf == 1)
    prfelv_lst = elv[prf==1]
    #print(prfelv_lst) [7, 71.1, 119.4]
    lat_coords = indices[0]
    lon_coords = indices[1]
    #print(x_coords, y_coords) > [648, 651, 652] [3834, 3831, 3830]

    # prf watershed
    rivnum_unq = np.unique(rivnum[prf == 1])
    cty_rivnum = [i for i in rivnum_unq]
    #print(cty_rivnum) # [848.0, 2718.0, 4850.0, 6065.0, 0]

    # city center data
    city_center = np.fromfile(cnt_path, dtype=dtype).reshape(lat_num, lon_num)

    # indices of city center
    indices = np.where(city_center==1) # tuple
    latcnt = int(indices[0])
    loncnt = int(indices[1])
    #print(x) #651
    #print(y) #3836

    # init maximum river discharge
    riv_max = 0

    # canal_out around 100km of city center
    can_mask = np.zeros((lat_num, lon_num))
    for ibt_lat in range(-exp_range, exp_range+1):
        for ibt_lon in range(-exp_range, exp_range+1):
            can_mask[latcnt+ibt_lat, loncnt+ibt_lon] = 1
    can_check = can_mask*can_out
    #print(np.sum(can_check)) # 0

    # display data
    display_data = np.zeros((lat_num, lon_num))

    # if canal exists
    if np.sum(can_check)>0:
        canal = 'canal_yes'
        if prfelv_lst.size == 0:
            print("no purification plant")
        else:

            # canal number
            canal_unq = np.unique(can_check)
            canal_lst = [uni for uni in canal_unq if uni>0]
            #print(canal_lst) # [100.]

            # canal unique number loop
            for canal_num in canal_lst:
                # indices of canal in
                can_ind = np.where(can_in==canal_num) # tuple
                can_ind = np.array(can_ind)
                #print(can_ind) # [[711, 711, 717], [2529, 2541, 2547]]
                #print(can_ind.shape) # (2, 3)

                # canal grid loop
                for C in range(can_ind.shape[1]):
                    display_data[can_out==can_check[can_ind[0, C],can_ind[1, C]]] = 1
                    # explore grids around canal
                    for p in range(-can_exp, can_exp):
                        for q in range(-can_exp, can_exp):
                            Y = can_ind[0, C] + p
                            X = can_ind[1, C] + q
                            display_data[Y, X] = 2
                            # maximum or not check
                            if riv_dis[Y,X]/1000. > riv_max:
                                # update riv
                                riv_max = riv_dis[Y,X]/1000.
                                YY = Y
                                XX = X

    # if no canal
    else:
        canal = 'canal_no'

        ### make search list
        search_lst = []
        for p in range(-exp_range, exp_range+1, 1):
            for q in range(-exp_range, exp_range+1, 1):
                    Y, X = latcnt + p, loncnt + q

                    # not explored yet
                    if 0 <= Y < lat_num and 0<= X < lon_num:
                        # distance btw prf and explored grid
                        d_list = []

                        for prf_y, prf_x in zip(lat_coords, lon_coords):
                            LON, LAT = xy2lonlat(X, Y)
                            prf_lon, prf_lat = xy2lonlat(prf_x, prf_y)
                            distance = lonlat_distance(LAT, LON, prf_lat, prf_lon)
                            d_list.append(distance)

                        # closer than IBT max distance
                        d_min = np.min(d_list)
                        elv_min = prfelv_lst[np.argmin(d_list)]

                        if d_min <= distance_condition:
                            search_lst.append([riv_dis[Y, X], Y, X])
                            display_data[Y, X] = 1

                            # out of city mask
                            if city_mask[Y, X] != 1:

                                # intake point shoud be higher than elevation of closest purification plant
                                if elv[Y, X] > elv_min:

                                    # including same watershed
                                    if no_prf_flag == 'True':
                                        display_data[Y, X] = 2

                                        # check if maximum
                                        if riv_dis[Y, X]/1000. > riv_max:
                                            # update riv
                                            riv_max = riv_dis[Y, X]/1000.
                                            #print(f'riv_max {X}, {Y} updated {riv_max}')
                                            YY = Y
                                            XX = X
                                            print(f"distance: {d_min}")

                                    # exclude same water shed
                                    else:
                                        # river num (watershed) is not overlapped with that of inner city
                                        if rivnum[Y, X] not in cty_rivnum:
                                            display_data[Y, X] = 2

                                            # check if maximum
                                            if riv_dis[Y, X]/1000. > riv_max:
                                                # update riv
                                                riv_max = riv_dis[Y, X]/1000.
                                                #print(f'riv_max {X}, {Y} updated {riv_max}')
                                                YY = Y
                                                XX = X
                                                print(f"distance: {d_min}")

    if riv_max > 0:

        # save file for display check
        display_data[YY, XX] =                   3
        #display_data[city_mask == 1] =           4
        #display_data[city_center == 1] =         5

        # save file for binary
        intake = np.zeros((lat_num, lon_num))
        intake[YY, XX] = 1
        print(f"riv_max  {riv_max}\n"
              f"{canal}")

    else:
        print("no potential intake point\n"
              f"riv_max  {riv_max}\n"
              f"{canal}")
        intake = np.zeros((lat_num, lon_num))

    # save
    if save_flag is True:
        intake.astype(np.float32).tofile(savepath)
        print(f"{savepath} saved")
        display_data.astype(np.float32).tofile(displaypath)
        print(f"{displaypath} saved")
    else:
        print(f"save_flag is {save_flag}")


def xy2lonlat(x, y, lat_num=2160, lon_num=4320):
    if 0 <= x <= lon_num:
        loncnt = (x*360/lon_num) - 180
        latcnt = 90 - (y*180)/lat_num
    else:
        loncnt = 1e20
        latcnt = 1e20

    return loncnt, latcnt



def lonlat_distance(lat_a, lon_a, lat_b, lon_b):
    """" Hybeny's Distance Formula """
    pole_radius = 6356752.314245
    equator_radius = 6378137.0
    radlat_a = math.radians(lat_a)
    radlon_a = math.radians(lon_a)
    radlat_b = math.radians(lat_b)
    radlon_b = math.radians(lon_b)

    lat_dif = radlat_a - radlat_b
    lon_dif = radlon_a - radlon_b
    lat_ave = (radlat_a + radlat_b) / 2

    e2 = (math.pow(equator_radius, 2) - math.pow(pole_radius, 2)) \
            / math.pow(equator_radius, 2)

    w = math.sqrt(1 - e2 * math.pow(math.sin(lat_ave), 2))

    m = equator_radius * (1 - e2) / math.pow(w, 3)

    n = equator_radius / w

    distance = math.sqrt(math.pow(m * lat_dif, 2) \
                + math.pow(n * lon_dif * math.cos(lat_ave), 2))

    return distance / 1000


def main():
    save_flag = True
    for city_num in range(1, 1861, 1):
        explore(city_num, save_flag)


if __name__ == '__main__':
    main()

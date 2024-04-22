import math
import numpy as np
import matplotlib.pyplot as plt

def explore_citymask(load_mask, index, err_count):

    #-----------------------------------------------
    # PATHS
    #-----------------------------------------------

    # suffix
    SUF = '30s'

    # pop data
    POP='gpw4_30s'

    # paths
    h08dir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city'
    wup_path = f'{h08dir}/dat/cty_lst_/gpw4/WUP2018_300k_2010.txt'
    area_path = f'/mnt/c/Users/tsimk/Downloads/H08_20230612/map/dat/grd_ara_/grdara.{SUF}'
    pop_path = f'{h08dir}/dat/pop_tot_/GPW4ag__20100000.{SUF}'
    center_path = f'{h08dir}/dat/cty_cnt_/{POP}/cityclrd0000.{SUF}'
    modified_center_path = f'{h08dir}/dat/cty_cnt_/{POP}/modified/cityclrd0000.{SUF}'
    result_text_path = h08dir + f'/dat/cty_lst_/{POP}/result_{SUF}.txt'
    save_mask_path = h08dir + f'/dat/cty_msk_/{POP}/city_clrd0000.{SUF}'

    #-----------------------------------------------
    # Input Constants
    #-----------------------------------------------

    # city center modification
    modify_flag = True

    # search radius (1grid in 30seconds)
    circle = 5

    # EN.1: lower limitation of population density
    lowlim = 1000

    # EN.2: initial grid threshold
    threshold = 1000

    # EN.3: grid sum
    grdlim = 1000

    # EN.3: downtown rate
    downtown_rate = 1.5

    # EN.4: low ratio
    lowrat = 0.0

    # shape
    lat_shape = 21600
    lon_shape = 43200

    # date type
    dtype= 'float32'

    #-----------------------------------------------
    # Initialization
    #-----------------------------------------------

    # initialize variables
    best_coverage = float('inf')
    best_mask = None
    best_masked_pop = None

    #-----------------------------------------------
    # load true data (UN city list) unit=[1000person]
    #-----------------------------------------------

    # true population and city name
    un_pop_list = []
    name_list = []

    # load data
    for l in open(wup_path).readlines():
        data = l[:].split('\t')
        data = [item.strip() for item in data]
        un_pop_list.append(float(data[3]))
        name_list.append(data[4])

    # get true UN city population
    un_pop = un_pop_list[index-1]*1000

    # get city name
    city_name = name_list[index-1]

    #-----------------------------------------------
    #  Get area(m2)
    #-----------------------------------------------

    area = np.fromfile(area_path, dtype=dtype).reshape(lat_shape, lon_shape)

    #-----------------------------------------------
    # load gwp population data
    #-----------------------------------------------

    # population data(GWP4 2010)
    gwp_pop = np.fromfile(pop_path, dtype=dtype).reshape(lat_shape, lon_shape)

    # mask ocean grid
    np.where(gwp_pop == 1e20, 0, gwp_pop)

    # population density (person/km2)
    gwp_pop_density = (gwp_pop / (area / 10**6))

    #-----------------------------------------------
    # check city center
    #-----------------------------------------------

    if modify_flag is True:
        center_path = f'{h08dir}/dat/cty_cnt_/{POP}/cityclrd0000.{SUF}'
        modified_center_path = f'{h08dir}/dat/cty_cnt_/{POP}/modified/cityclrd0000.{SUF}'

        location = np.fromfile(center_path, dtype=dtype).reshape(lat_shape,lon_shape)
        org_y = np.where(location==index)[0]
        org_x = np.where(location==index)[1]
        org_y = org_y[0]
        org_x = org_x[0]

        # modified city center
        modified = np.fromfile(modified_center_path, dtype=dtype).reshape(lat_shape,lon_shape)

        # original city center
        org_cnt = gwp_pop_density[org_y, org_x]

        # number of replacement
        replaced_num = 0
        print(f"cityindex {index}")
        print(f'original center [y, x] = [{org_y, org_x}]')
        print(f"org_cnt: {org_cnt}")

        # if there is larger grid, center grid is replaced
        rpl_y, rpl_x = org_y, org_x
        for a_cnt in range(org_y-circle, org_y+circle+1):
            for b_cnt in range(org_x-circle, org_x+circle+1):
                candidate = gwp_pop_density[a_cnt, b_cnt]
                if candidate >= org_cnt:
                    org_cnt = candidate
                    rpl_y = a_cnt
                    rpl_x = b_cnt
                    replaced_num += 1

        print(f'replaced center [y, x] = [{rpl_y, rpl_x}]')
        print(f"rpl_cnt: {gwp_pop_density[rpl_y, rpl_x]}")

        modified[org_y, org_x] = 0
        modified[rpl_y, rpl_x] = index
        modified.astype(np.float32).tofile(modified_center_path)

    else:
        modified_center_path = f'{h08dir}/dat/cty_cnt_/{POP}/modified/cityclrd0000.{SUF}'

    mod_y = np.where(modified==index)[0]
    mod_x = np.where(modified==index)[1]
    mod_y = mod_y[0]
    mod_x = mod_x[0]

    #-----------------------------------------------
    #  Initialization of mask array
    #-----------------------------------------------

    # mask array for saving
    mask = np.zeros((lat_shape,lon_shape), dtype=dtype)
    mask[mod_y, mod_x] = 1

    #-----------------------------------------------
    #  Explore start
    #-----------------------------------------------

    # no err
    err_flag = 0

    # stop flag
    new_mask_added = True
    coverage_flag = True

    # city center
    best_mask = mask
    grid_num = np.sum(best_mask)
    best_masked_pop = np.sum(gwp_pop*mask)
    best_coverage = float(best_masked_pop / un_pop)

    # momnitor density ratio
    init_density = np.sum(gwp_pop_density[mod_y, mod_x])
    previous_density = np.sum(gwp_pop_density[mod_y, mod_x])

    # initial grid threshold
    if gwp_pop_density[mod_y, mod_x] <= threshold:
        print("----------/// 111 ///----------")
        print(f"initial density {gwp_pop_density[mod_y, mod_x]} less than threshold {threshold}")
        print("----------/// 111 ///----------")
        new_mask_added = False
        coverage_flag = False
        density_ratio = (previous_density/init_density)*100
        err_flag = 1
        best_mask = np.zeros((lat_shape,lon_shape), dtype=dtype)

    # loop start
    while new_mask_added:

        ### make search list
        search_lst = []
        new_mask_added = False
        indices = np.where(mask == 1)

        for ind in range(len(indices[0])):
            y_index = indices[0][ind]
            x_index = indices[1][ind]
            for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, 1)]:
                i = y_index + dy
                j = x_index + dx
                # not explored yet
                if mask[i, j] == 0:
                    # within grid range
                    if 0 <= i < lat_shape and 0<= j < lon_shape:
                        # Don't use gwp_pop. there is bug due to ocean land ara data
                        search_lst.append([gwp_pop_density[i, j], i, j])

        ### obtain largest searched grid
        # empty check
        if not search_lst:

            new_mask_added = False
            coverage_flag = False
            print(f"search_lst is empty")

        # get largest grid
        else:

            sorted_search = sorted(search_lst, key=lambda x: x[0], reverse=True)
            largest = sorted_search[0]
            next_density = gwp_pop_density[largest[1], largest[2]]
            density_ratio = (next_density/init_density)*100

            # if largest grid in searched grid is too small, stop exploring
            if next_density <= lowlim:
                print("----------/// 222 ///----------")
                print(f"next density {next_density} smaller than lowlim {lowlim}")
                print("----------/// 222 ///----------")
                new_mask_added = False
                coverage_flag = False
                err_flag = 2

            elif next_density > previous_density and best_coverage > downtown_rate and grid_num >= grdlim:
                print("----------/// 333 ///----------")
                print(f"next density {next_density} bigger than previous density {previous_density}")
                print("----------/// 333 ///----------")
                new_mask_added = False
                coverage_flag = False
                err_flag = 3

            elif density_ratio < lowrat:
                print("----------/// 444 ///----------")
                print(f"next density {next_density} less than one-tenth of initial density {init_density}")
                print("----------/// 444 ///----------")
                new_mask_added = False
                coverage_flag = False
                err_flag = 4


        ### add new mask
        # stop flag
        if coverage_flag is True:
                # new mask added
                mask[largest[1], largest[2]] = 1
                new_mask_added = True

                # evaluate coverage
                gwp_masked_pop = np.sum(mask * gwp_pop)
                coverage = float(gwp_masked_pop / un_pop)

                # stop exploring
                if coverage >= 1.0:
                    new_mask_added = False
                    coverage_flag = False

                # judge
                judge_value = abs(1 - coverage)
                best_value = abs(1 - best_coverage)

                # update
                if judge_value < best_value:
                    best_coverage = coverage
                    best_mask = mask
                    best_masked_pop = gwp_masked_pop
                    grid_num = np.sum(best_mask)
                    previous_density = gwp_pop_density[largest[1], largest[2]]
                    print(f"city index={index}:, cover: {best_coverage}, num: {grid_num}")
    #-----------------------------------------------
    #  overlap check
    #-----------------------------------------------

    if np.any(np.logical_and(load_mask > 0, best_mask > 0)):
        print("----------/// 555 ///----------")
        print(f"overlap occured")
        print("----------/// 555 ///----------")
        new_mask_added = False
        coverage_flag = False
        err_flag = 5
        best_mask = np.zeros((lat_shape,lon_shape), dtype=dtype)

    #-----------------------------------------------
    # Output result
    #-----------------------------------------------

    print(
          f"explored_pop {best_masked_pop}\n" \
          f"true_pop {un_pop}\n" \
          f"coverage {best_coverage}\n" \
          f"city_mask {grid_num}\n" \
          f"density_ratio {density_ratio}\n" \
          f"err_flag {err_flag}\n" \
          f"{city_name}\n"
          )
    print('#########################################\n')

    #------------------------------------------------
    # SAVE FILE
    #------------------------------------------------

    if index == 1:
        with open(result_text_path, 'w') as file:
            file.write(f"{index}| {city_name}| {best_masked_pop}| {un_pop}| {best_coverage}| {grid_num}| {err_flag}\n")
    else:
        with open(result_text_path, 'a') as file:
            file.write(f"{index}| {city_name}| {best_masked_pop}| {un_pop}| {best_coverage}| {grid_num}| {err_flag}\n")

    # update error
    err_count[f'{err_flag}'] += 1
    print(err_count)

    # mask binary file saved
    load_mask[best_mask == 1] == index
    load_mask.astype(np.float32).tofile(save_mask_path)

    return load_mask, err_count


def main():
    # initial state
    dtype= 'float32'
    lat_shape = 21600
    lon_shape = 43200
    load_mask = np.zeros((lat_shape,lon_shape), dtype=dtype)
    err_count = {'0': 0, '1': 0, '2': 0, '3':0, '4':0, '5':0}

    # python make_downtown.py > make_downtown.log
    for index in range(1, 1861):
        load_mask, err_count = explore_citymask(load_mask, index, err_count)


if __name__ == '__main__':
    main()

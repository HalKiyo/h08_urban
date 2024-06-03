import os
import pickle
import numpy as np

def clustering(index, save_dict):

    #---------------------------------------------------------------
    # PATHS
    #---------------------------------------------------------------

    # paths
    rootdir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city'
    load_path = f'{rootdir}/dat/dwn_twn_/city_{index:08d}.pickle'
    mask_path = f'{rootdir}/dat/vld_cty_/city_{index:08d}.gl5'

    #---------------------------------------------------------------
    # Input Constants
    #---------------------------------------------------------------

    # EN.1: lower limitation of population density
    lowlim = 100

    # EN.2 initial grid threshold
    threshold = 300

    # EN.3 Downtown positive gradient continuity
    pg_range = 3

    # EN.3 Minimum city mask for downtown algorithm
    gridlim = 10

    # EN.4 Minimum city mask
    mingrid = 3


    #---------------------------------------------------------------
    # Load Pickle
    #---------------------------------------------------------------

    with open (load_path, 'rb') as file:
        load_dict = pickle.load(file)

    #---------------------------------------------------------------
    # Load Density_track
    #---------------------------------------------------------------

    density_track = load_dict['added_density']

    #---------------------------------------------------------------
    # Valid or Invalid Mask
    #---------------------------------------------------------------

    if not density_track:
        print(f"city_index: {index}")
        print(f"invalid mask")
        save_dict['gradient'].append(-1)
        save_dict['mask_num'].append(-1)
        save_dict['cover_rate'].append(-1)
        save_dict['invalid_index'].append(index)
        return save_dict

    if len(density_track) == 1:
        print(f"city_index: {index}")
        print(f"initial_density: {density_track[0]}")
        print(f"invalid mask")
        save_dict['gradient'].append(-1)
        save_dict['mask_num'].append(-1)
        save_dict['cover_rate'].append(-1)
        save_dict['invalid_index'].append(index)
        return save_dict

    if density_track[0] <=  threshold:
        print(f"city_index: {index}")
        print(f"initial_density: {density_track[0]}")
        print(f"invalid mask")
        save_dict['gradient'].append(-1)
        save_dict['mask_num'].append(-1)
        save_dict['cover_rate'].append(-1)
        save_dict['invalid_index'].append(index)
        return save_dict

    #---------------------------------------------------------------
    # Valid Gradient
    #---------------------------------------------------------------

    if density_track:
        pg_index = []
        for d in range(len(density_track)-1):
            if density_track[d+1] > density_track[d]:
                pg_index.append(d)
        positive_gradient = [density_track[p] for p in pg_index]

        valid_index = []
        for p in range(len(pg_index)-1):
            if (pg_index[p] + 1) >= gridlim:
                if (pg_index[p+1] - pg_index[p]) <= pg_range:
                    valid_index.append(pg_index[p])
        valid_gradient = [density_track[v] for v in valid_index]

    #---------------------------------------------------------------
    # Downtown algorithm
    #---------------------------------------------------------------

    if valid_index:
        target_index = valid_index[0]
        gradient = valid_gradient[0]
        bestmask_track = load_dict['mask'][:, :, target_index]
        mask_cover = load_dict['cover_rate'][target_index]
        mask_num = np.sum(bestmask_track)
        if mask_num >= mingrid:
            print(f"city_index: {index}")
            print(valid_index)
            print(f"len of positive_gradient: {len(positive_gradient)}")
            print(f"len of valid_gradient: {len(valid_gradient)}")
            print(f"gradient: {gradient}")
            print(f"city_mask: {mask_num}")
            print(f"mask_cover: {mask_cover}")
            save_dict['gradient'].append(gradient)
            save_dict['mask_num'].append(mask_num)
            save_dict['cover_rate'].append(mask_cover)
            bestmask_track.astype(np.float32).tofile(mask_path)
        else:
            print(f"city_index: {index}")
            print(f"gradient: {gradient}")
            print(f"city_mask: {mask_num}")
            print(f"mask_cover: {mask_cover}")
            print(f"invalid mask")
            save_dict['gradient'].append(gradient)
            save_dict['mask_num'].append(mask_num)
            save_dict['cover_rate'].append(mask_cover)
            save_dict['invalid_index'].append(index)

    else:
        target_index = len(density_track) - 1
        gradient = density_track[target_index]
        bestmask_track = load_dict['mask'][:, :, target_index]
        mask_cover = load_dict['cover_rate'][target_index]
        mask_num = np.sum(bestmask_track)
        if mask_num >= mingrid:
            print(f"city_index: {index}")
            print(f"gradient: {gradient}")
            print(f"city_mask: {mask_num}")
            print(f"mask_cover: {mask_cover}")
            save_dict['gradient'].append(gradient)
            save_dict['mask_num'].append(mask_num)
            save_dict['cover_rate'].append(mask_cover)
            bestmask_track.astype(np.float32).tofile(mask_path)
        else:
            print(f"city_index: {index}")
            print(f"gradient: {gradient}")
            print(f"city_mask: {mask_num}")
            print(f"mask_cover: {mask_cover}")
            print(f"invalid mask")
            save_dict['gradient'].append(gradient)
            save_dict['mask_num'].append(mask_num)
            save_dict['cover_rate'].append(mask_cover)
            save_dict['invalid_index'].append(index)

    return save_dict


#-----------------------------------------------------------------------

def main():
    """
    loop 1860 cities in WUP
    clustering returns {gradient, mask_num, cover_rate, invalid_index}
    if A - B <= pg_range, then that's valid_index
    if invalid, then gradient, mask_num, cover_rate = -1, -1, -1
    """
    save_dict = {'gradient': [],
                 'mask_num': [],
                 'cover_rate': [],
                 'invalid_index': []}

    for city_index in range(1, 1861):
        save_dict = clustering(city_index, save_dict)
        print(f"invalid number {len(save_dict['invalid_index'])}")
        print(f"--------------------------------------------------------------")

    # paths
    rootdir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city'
    save_path = f'{rootdir}/dat/vld_cty_/city_00000000.pickle'

    # dict save
    with open(save_path, 'wb') as handle:
        pickle.dump(save_dict, handle)

#-----------------------------------------------------------------------

def summarize():

    # colored
    color_flag = True
    #
    POP = 'vld_cty_'
    #
    lat_shape = 2160
    lon_shape = 4320
    dtype= 'float32'

    # paths
    h08dir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city'
    color_path = f"{h08dir}/dat/{POP}/city_clrd0000.gl5"
    monochrome_path = f"{h08dir}/dat/{POP}/city_00000000.gl5"
    ovlp_color_path = f"{h08dir}/dat/{POP}/city_clrdovlp.gl5"
    ovlp_monochrome_path = f"{h08dir}/dat/{POP}/city_0000ovlp.gl5"
    # paths
    wup_path = f'{h08dir}/dat/cty_lst_/gpw4/WUP2018_300k_2010.txt'
    textpath = h08dir + f'/dat/cty_lst_/gpw4/downtown_overlap.txt'

    # city_name
    name_list =  []
    pop_list = []
    for l in open(wup_path).readlines():
        data = l[:].split('\t')
        data = [item.strip() for item in data]
        pop_list.append(float(data[3]))
        name_list.append(data[4])

    # overlap text
    with open(textpath, 'w') as file:
        file.write(f"index | pop | grid_num | name|\n")

    # shape
    lat_shape = 2160
    lon_shape = 4320

    # date type
    dtype= 'float32'

    # make save array
    summary = np.empty((lat_shape, lon_shape))
    overlap = np.empty((lat_shape, lon_shape))

    # city index loop
    for index in range(1, 1861):
        city_name = name_list[index-1]
        city_pop = pop_list[index-1]

        # load city mask
        mask_path = f"{h08dir}/dat/{POP}/city_{index:08}.gl5"

        # mask existance
        if not os.path.exists(mask_path):
            print(f'{index} is invalid mask')
            save_path = None
            ovlp_save_path = None
        else:
            tmp = np.fromfile(mask_path, dtype=dtype).reshape(lat_shape, lon_shape)
            # overlap check
            if np.sum(summary[tmp==1]) < 1:
                # color or monotchrome
                if color_flag is True:
                    summary[tmp == 1] = index
                    save_path = color_path
                else:
                    summary[tmp == 1] = 1
                    save_path = monochrome_path
                print(f'{index} is valid mask')
            else:
                if color_flag is True:
                    overlap[tmp == 1] = index
                    ovlp_save_path = ovlp_color_path
                    save_path = ovlp_color_path
                else:
                    overlap[tmp == 1] = 1
                    ovlp_save_path = ovlp_monochrome_path

                with open(textpath, 'a') as file:
                    file.write(f"{index}| {city_pop}| {np.sum(tmp)}| {city_name}\n")

                print(f'{index} is overlaped')

    summary.astype(np.float32).tofile(save_path)
    overlap.astype(np.float32).tofile(ovlp_save_path)
    print(f'{save_path} is saved')

#-----------------------------------------------------------------------

def check():
    """
    save_dict = {'gradient': [],
                 'mask_num': [],
                 'cover_rate': [],
                 'invalid_index': []}
    """

    # paths
    rootdir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city'
    save_path = f'{rootdir}/dat/vld_cty_/city_00000000.pickle'

    with open(save_path, 'rb') as file:
        save_dict = pickle.load(file)

    invalid_index = save_dict['invalid_index']

    for index in range(1600, 1861):
        if index not in invalid_index:
            mask_path = f'{rootdir}/dat/vld_cty_/city_{index:08d}.gl5'
            print(f"index: {index}")
            print(f"gradient: {save_dict['gradient'][index-1]}")
            print(f"mask_num: {save_dict['mask_num'][index-1]}")
            print(f"cover_rate: {save_dict['cover_rate'][index-1]}")
            print('-------------------------------------')

    print(invalid_index)
    print(f"len(invalid_index): {len(invalid_index)}")

#-----------------------------------------------------------------------

if __name__ == '__main__':
    #main()
    summarize()
    #check()


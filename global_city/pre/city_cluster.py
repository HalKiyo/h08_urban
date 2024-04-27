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
    #mask_path = f'{rootdir}/dat/vld_cty_/city_{index:08d}.gl5'

    for index in range(1, 200):
        if index not in invalid_index:
            print(f"index: {index}")
            print(f"gradient: {save_dict['gradient'][index]}")
            print(f"mask_num: {save_dict['mask_num'][index]}")
            print(f"cover_rate: {save_dict['cover_rate'][index]}")
            print('-------------------------------------')


if __name__ == '__main__':
    #main()
    check()


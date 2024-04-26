import pickle
import numpy as np

def clustering(index):

    #---------------------------------------------------------------
    # PATHS
    #---------------------------------------------------------------

    # paths
    rootdir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city'
    dict_path = f'{rootdir}/dat/dwn_twn_/city_{index:08d}.pickle'

    #---------------------------------------------------------------
    # Input Constants
    #---------------------------------------------------------------

    # EN.1: lower limitation of population density
    lowlim = 300
    # これも作動させる必要があるかも

    # EN.2 initial grid threshold
    threshold = 500

    # EN.3 Downtown positive gradient continuity
    pg_range = 3

    # EN.3 Minimum city mask
    gridlim = 10
    # これでvalid_indexを制御しないとrio de janeiroとかでcity mask=2とかになってしまう

    #---------------------------------------------------------------
    # Load Pickle
    #---------------------------------------------------------------

    with open (dict_path, 'rb') as file:
        save_dict = pickle.load(file)

    #---------------------------------------------------------------
    # Valid Gradient
    #---------------------------------------------------------------

    density_track = save_dict['added_density']

    pg_index = []
    for d in range(len(density_track)-1):
        if density_track[d+1] > density_track[d]:
            pg_index.append(d)
    positive_gradient = [density_track[p] for p in pg_index]

    valid_index = []
    for p in range(len(pg_index)-1):
        if (pg_index[p+1] - pg_index[p]) <= pg_range:
            valid_index.append(pg_index[p])
    valid_gradient = [density_track[v] for v in valid_index]

    #---------------------------------------------------------------
    # Down-town Mask
    #---------------------------------------------------------------

    if valid_index:
        target_index = valid_index[0]
        gradient = valid_gradient[0]
    else:
        target_index = len(density_track) - 1
        gradient = density_track[target_index]
    bestmask_track = save_dict['mask'][:, :, target_index]
    mask_cover = save_dict['cover_rate'][target_index]
    mask_num = np.sum(bestmask_track)

    #---------------------------------------------------------------
    # Check
    #---------------------------------------------------------------

    print(f"city_index: {index}")
    print(f"len of positive_gradient: {len(positive_gradient)}")
    print(f"len of valid_gradient: {len(valid_gradient)}")
    print(f"gradient: {gradient}")
    print(f"city_mask: {mask_num}")
    print(f"mask_cover: {mask_cover}")
    print("--------------------------")

def main():
    for city_index in range(1, 1861):
        clustering(city_index)


if __name__ == '__main__':
    main()


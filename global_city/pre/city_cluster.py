import pickle
import numpy as np

def clustering(index, invalid_index):

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

    with open (dict_path, 'rb') as file:
        save_dict = pickle.load(file)

    #---------------------------------------------------------------
    # Load Density_track
    #---------------------------------------------------------------

    density_track = save_dict['added_density']

    #---------------------------------------------------------------
    # Valid or Invalid Mask
    #---------------------------------------------------------------

    if not density_track:
        print(f"city_index: {index}")
        print(f"invalid mask")
        print("--------------------------")
        invalid_index.append(index)
        return invalid_index

    if len(density_track) == 1:
        print(f"city_index: {index}")
        print(f"initial_density: {density_track[0]}")
        print(f"invalid mask")
        print("--------------------------")
        invalid_index.append(index)
        return invalid_index

    if density_track[0] <=  threshold:
        print(f"city_index: {index}")
        print(f"initial_density: {density_track[0]}")
        print(f"invalid mask")
        print("--------------------------")
        invalid_index.append(index)
        return invalid_index

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
        print(valid_index)
        target_index = valid_index[0]
        gradient = valid_gradient[0]
        bestmask_track = save_dict['mask'][:, :, target_index]
        mask_cover = save_dict['cover_rate'][target_index]
        mask_num = np.sum(bestmask_track)
        if mask_num >= mingrid:
            print(f"city_index: {index}")
            print(f"len of positive_gradient: {len(positive_gradient)}")
            print(f"len of valid_gradient: {len(valid_gradient)}")
            print(f"gradient: {gradient}")
            print(f"city_mask: {mask_num}")
            print(f"mask_cover: {mask_cover}")
            print("--------------------------")
        else:
            print(f"city_index: {index}")
            print(f"gradient: {gradient}")
            print(f"city_mask: {mask_num}")
            print(f"mask_cover: {mask_cover}")
            print(f"invalid mask")
            print("--------------------------")
            invalid_index.append(index)

    else:
        target_index = len(density_track) - 1
        gradient = density_track[target_index]
        bestmask_track = save_dict['mask'][:, :, target_index]
        mask_cover = save_dict['cover_rate'][target_index]
        mask_num = np.sum(bestmask_track)
        if mask_num >= mingrid:
            print(f"city_index: {index}")
            print(f"gradient: {gradient}")
            print(f"city_mask: {mask_num}")
            print(f"mask_cover: {mask_cover}")
            print("--------------------------")
        else:
            print(f"city_index: {index}")
            print(f"gradient: {gradient}")
            print(f"city_mask: {mask_num}")
            print(f"mask_cover: {mask_cover}")
            print(f"invalid mask")
            print("--------------------------")
            invalid_index.append(index)

    return invalid_index


#-----------------------------------------------------------------------

def main():
    invalid_index = []
    for city_index in range(24, 1861):
        invalid_index = clustering(city_index, invalid_index)
        print(f"invalid number {len(invalid_index)}")


if __name__ == '__main__':
    main()


import os
import numpy as np

def main():
    # change the flag to True to overwrite
    save_flag = True

    # basic information
    tag = '.pr5'
    city_num = 21
    #
    resolution=12
    SUF ='.gl5'
    dtype = 'float32'
    gl5shape = (2160, 4320)
    #
    left = 0
    right = 6
    bottom = 46
    top = 52
    #
    nx, ny = (right - left)*resolution, (top - bottom)*resolution
    rgnshape = (ny, nx)
    img_extent = (left, right, bottom, top)
    #
    upperindex = (90-top)*resolution
    lowerindex = (90-bottom)*resolution
    leftindex = (180+left)*resolution
    rightindex = (180+right)*resolution
    #
    glbdir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city'
    rgndir = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/region'
    h08dir = '/mnt/c/Users/tsimk/Downloads/H08_20230612'
    #
    varlist = ['LWdown__', 'PSurf___', 'Rainf___', 'SWdown__', 'Wind____', 'Prcp____', 'Qair____', 'Snowf___', 'Tair____']

    # Job
    for var in varlist:
        metdir = h08dir + '/met/dat/' + var
        search_word1 = 'W5E5____2019'
        search_word2 = '.gl5'
        matching_files = find_files_with_word_in_filename(metdir, search_word1, search_word2)
        for file in matching_files:
            loadfile = file
            savefile = file.replace(SUF, tag)
            savefile = savefile.replace('H08_20230612/met', 'dotfiles/h08/region')

            print(loadfile)
            data = np.fromfile(loadfile, dtype=dtype).reshape(gl5shape)
            tokyo = data[upperindex:lowerindex, leftindex:rightindex]
            if save_flag is True:
                tokyo.astype(np.float32).tofile(savefile)
                print(f"{savefile} is saved")
        print(f"{var} done")

def find_files_with_word_in_filename(directory, word1, word2):
    """
    directory_path = '/path/to/your/directory'
    search_word1 = 'target_word1'
    search_word2 = 'target_word2'
    """

    matching_files = []

    # obtain all file in the directory
    files = os.listdir(directory)

    # check each file
    for file in files:
        # files should contain both words
        if word1 in file and word2 in file:
            file_path = os.path.join(directory, file)
            matching_files.append(file_path)

    return matching_files


if __name__ == '__main__':
    main()

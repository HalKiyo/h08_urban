import numpy as np
import rasterio
from rasterio.transform import from_origin
 
def bin2las(loadpath, savepath):
    data = np.fromfile(loadpath, dtype='float32')
    data = data.reshape([2160, 4320])

    #data[data <= 10000] = 0

    west = -180
    north = 90
    transform = from_origin(west, north, 5/60, 5/60)

    with rasterio.open(savepath, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, dtype=str(data.dtype), crs='+proj=latlong', transform=transform) as dst:
       dst.nodata = 0 
       dst.write(data, 1)
    print(f"{savepath} is saved")

def main():
    """
    loadpath = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/dat/cty_swg_/gpw3/city_00000000.gl5'
    savepath = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/qgis/dat/cty_msk_gpw3_00000000.tif'
    """
    loadpath = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/camacity/dat/cty_msk_/city_clrd0000.gl5'
    savepath = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/qgis/dat/camacity_clrd0000.tif'
    bin2las(loadpath, savepath)

if __name__ == '__main__':
    main()

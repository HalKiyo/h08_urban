import numpy as np
import rasterio
from rasterio.transform import from_origin
 
def bin2las(loadpath, savepath):
    # バイナリデータの読み込みと変換
    data = np.fromfile(loadpath, dtype='float32')
    data = data.reshape([2160, 4320]) # データの行数と列数

    # 10000以下の値をnodataに置き換える
    #data[data <= 10000] = 0

    # ラスタデータの変換パラメータを設定
    west = -180 # 西経
    north = 90 # 北緯
    transform = from_origin(west, north, 5/60, 5/60)

    # 新しいGeoTIFFファイルを作成
    with rasterio.open(savepath, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, dtype=str(data.dtype), crs='+proj=latlong', transform=transform) as dst:
       dst.nodata = 0 # nodata値を0に設定
       dst.write(data, 1)
    print(f"{savepath} is saved")

def main():
    """
    loadpath = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/dat/cty_swg_/gpw3/city_00000000.gl5'
    savepath = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/qgis/dat/cty_msk_gpw3_00000000.tif'
    """
    loadpath = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/dat/cty_swg_/gpw4/city_00000001.gl5'
    savepath = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/qgis/dat/cty_swg_00000001.tif'
    bin2las(loadpath, savepath)

if __name__ == '__main__':
    main()

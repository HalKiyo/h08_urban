import numpy as np

def lonlat2xy(n0x, n0y, p0lonmin, p0lonmax, p0latmin, p0latmax, r1dat, r0dat, r0lon, r0lat):
    """
    cnt   fortran starts from 1 but python starts 0 then adding 0.5 is removed from original igeti0x.f
    cby   2010/03/31, hanasaki, NIES: H08 ver1.0
    c
    c       (plonmin,platmin)
    c         /
    c        /________   __________
    c        |      |       |       |
    c        |(0, 0)|  ...  |(nx, 0)|
    c        |______|__   __|_______|
    c        |      |               |
    c           .     .
    c           .       .
    c           .         .
    c        |______|__   __________|
    c        |      |       |       |
    c        |(0,ny)|  ...  |(nx,ny)|
    c        |______|__   __|_______|
    c
    """

    # load file
    n0x = int(n0x)
    n0y = int(n0y)
    p0lonmin = float(p0lonmin)
    p0lonmax = float(p0lonmax)
    p0latmin = float(p0latmin)
    p0latmax = float(p0latmax)
    r0dat = float(r0dat)
    r0lon = float(r0lon)
    r0lat = float(r0lat)

    if p0lonmin <= r0lon <= p0lonmax:
        i0x = int((r0lon - p0lonmin)/(p0lonmax - p0lonmin) * float(n0x))
    else:
        i0x = 1e20

    if p0latmin <= r0lat <= p0latmax:
        i0y = int((p0latmax - r0lat)/(p0latmax-p0latmin) * float(n0y))
    else:
        i0y = 1e20

    r1dat[i0y, i0x] = r0dat

    return r1dat

def main():
    n0x = 43200
    n0y = 21600
    p0lonmin = -180
    p0lonmax = 180
    p0latmin = -90
    p0latmax = 90

    WUPPATH = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/dat/cty_lst_/gpw4/WUP2018_300k_2010.txt'
    OUTPATH = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/dat/cty_cnt_/gpw4_30s/cityclrd0000.30s'

    tmpfile = np.zeros((n0y, n0x))

    index = 1
    for l in open(WUPPATH).readlines():
        data = l[:].split('\t')
        data = [item.strip() for item in data]
        r0lat = float(data[1])
        r0lon = float(data[2])
        tmpfile = lonlat2xy(n0x, n0y, p0lonmin, p0lonmax, p0latmin, p0latmax, tmpfile, index, r0lon, r0lat)
        index += 1

    tmpfile.astype(np.float32).tofile(OUTPATH)

if __name__ == '__main__':
    main()

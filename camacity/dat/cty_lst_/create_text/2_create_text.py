import os
import numpy as np

#########################################################
# city_index

# downtown flag
# overlap flag

# wup pop
# estimated 100% pop
# valid city estimated pop

# region
# country
# lat lon
# cityname
#########################################################

def main():
    cama_dir    = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/camacity'
    glob_dir    = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city'

    wup_path    = f"{cama_dir}/dat/cty_lst_/wup_vs_vldcty.txt"
    full_path   = f"{cama_dir}/dat/cty_lst_/result_downtown_100percent.txt"
    latlon_path = f"{glob_dir}/dat/cty_lst_/gpw4/WUP2018_300k_2010_regionadded.txt"

    pop_path    = f'{glob_dir}/dat/pop_tot_/GPW4ag__20100000.gl5'
    vldmsk_dir  = f'{glob_dir}/dat/vld_cty_'
    modmsk_dir  = f'{cama_dir}/dat/dwn_msk_'
    lowmsk_dir  = f'{cama_dir}/dat/low_msk_'
    ovlmsk_dir  = f'{cama_dir}/dat/ovlpmsk_'

    save_path   = f"{cama_dir}/dat/cty_lst_/create_text/camacity_second.txt"

    wup = open_text(wup_path)
    full = open_text(full_path)
    latlon = open_text(latlon_path)

    gpw = np.fromfile(pop_path, dtype='float32').reshape(2160, 4320)

    for ind in range(1860):
        vld_path = f'{vldmsk_dir}/city_{ind+1:08}.gl5'
        mod_path = f'{modmsk_dir}/city_kj_{ind+1:08}.gl5'
        low_path = f'{lowmsk_dir}/city_kj_{ind+1:08}.gl5'
        ovl_path = f'{ovlmsk_dir}/city_kj_{ind+1:08}.gl5'

        # dwn_flg decision
        if not os.path.exists(vld_path):
            if not os.path.exists(low_path):
                vld_pop = 'NA'
                msk_path = 'NA'
                grid_num = 'NA'
                dwn_flg = 'NoMK'
            else:
                msk_path = low_path
                dwn_flg = 'SMLL'
        elif os.path.exists(mod_path):
            msk_path = mod_path
            dwn_flg = 'DOWN'
        else:
            msk_path = vld_path
            dwn_flg = 'FULL'

        # ovlp_flg decision
        if not os.path.exists(ovl_path):
            ovlp_flg = 'VALD'
        else:
            msk_path = ovl_path
            ovlp_flg = 'OVLP'

        # calc valid pop
        if msk_path != 'NA':
            print(msk_path)
            mask = np.fromfile(msk_path, dtype='float32').reshape(2160, 4320)
            vld_pop = np.sum(gpw[mask!=0])
            grid_num = np.sum(mask)
            if grid_num == 0:
                ovlp_flg = 'RMVD'

        # latlon text
        line_latlon = latlon[ind]
        parts_latlon = line_latlon.split('\t')
        parts_latlon = [item.strip() for item in parts_latlon]
        lat = float(parts_latlon[0])
        lon = float(parts_latlon[1])

        # wup and vld_cty_ pop country region cityname text
        line_wup  = wup[ind]
        parts_wup = line_wup.split('|')
        parts_wup = [item.strip() for item in parts_wup]
        wup_pop = float(parts_wup[2])
        country = parts_wup[4]
        region = parts_wup[5]
        city_name = parts_wup[6].replace("\"", "").replace("?", "").replace("/", "")

        # full pop text
        line_full  = full[ind]
        parts_full = line_full.split('|')
        parts_full = [item.strip() for item in parts_full]
        fll_pop  = float(parts_full[2])

        write_text(save_path, ind, ovlp_flg, dwn_flg, wup_pop, fll_pop, vld_pop, region, grid_num, country, lat, lon, city_name)
        print(f'{ind} done')

def write_text(save_path, index, ovlp_flg, dwn_flg, wup_pop, fll_pop, vld_pop, region, grid_num, country, lat, lon, city_name):
    if index == 0:
        with open(save_path, 'w') as file:
            file.write(f"{index+1}|{ovlp_flg}|{dwn_flg}|{wup_pop}|{fll_pop}|{vld_pop}|{region}|{grid_num}|{country}|{lat}|{lon}|{city_name}\n")
    else:
        with open(save_path, 'a') as file:
            file.write(f"{index+1}|{ovlp_flg}|{dwn_flg}|{wup_pop}|{fll_pop}|{vld_pop}|{region}|{grid_num}|{country}|{lat}|{lon}|{city_name}\n")

def open_text(path):
    with open(path, 'r') as files:
        data = files.readlines()
    return data


if __name__ == '__main__':
    main()

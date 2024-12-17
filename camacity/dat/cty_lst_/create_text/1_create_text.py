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
    wup_path    = f"{cama_dir}/dat/cty_lst_/wup_vs_vldcty.txt"
    full_path   = f"{cama_dir}/dat/cty_lst_/result_downtown_100percent.txt"
    dwnflg_path = f"{cama_dir}/dat/cty_lst_/cluster_result.txt"
    ovlflg_path = f"{cama_dir}/dat/cty_lst_/vld_cty_overlap.txt"
    save_path   = f"{cama_dir}/dat/cty_lst_/camacity_first.txt"

    glob_dir    = '/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city'
    latlon_path = f"{glob_dir}/dat/cty_lst_/gpw4/WUP2018_300k_2010_regionadded.txt"

    wup = open_text(wup_path)
    full = open_text(full_path)
    dwn = open_text(dwnflg_path)
    latlon = open_text(latlon_path)
    over = open_text(ovlflg_path)

    over_list = []
    for line_over in over:
        columns = line_over.split('|')
        index_value = columns[0].strip()
        over_list.append(int(index_value))

    for ind in range(1860):
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
        ovl_flg = parts_wup[1]
        wup_pop = float(parts_wup[2])
        country = parts_wup[4]
        region = parts_wup[5]
        city_name = parts_wup[6].replace("\"", "").replace("?", "").replace("/", "")

        # full pop text
        line_full  = full[ind]
        parts_full = line_full.split('|')
        parts_full = [item.strip() for item in parts_full]
        fll_pop  = float(parts_full[2])

        # dwntwn flag text
        line_dwn  = dwn[ind]
        parts_dwn = line_dwn.split('|')
        parts_dwn = [item.strip() for item in parts_dwn]
        dwn_flg  = parts_dwn[1]

        # flag classfication
        if ind+1 in over_list:
            ovl_flg = 'OVERLAP'

        if ovl_flg != 'NoMASK':
            if ind+1 in over_list:
                ovl_flg = 'OVERLAP'
                vld_pop = float(parts_wup[3])
            else:
                ovl_flg = 'VALID'
                vld_pop = float(parts_wup[3])
        else:
            ovl_flg = 'NoMASK'
            vld_pop = 'NA'

        write_text(save_path, ind, ovl_flg, dwn_flg, wup_pop, fll_pop, vld_pop, country, region, lat, lon, city_name)

def write_text(save_path, index, dwn_flg, ovl_flg, wup_pop, fll_pop, vld_pop, country, region, lat, lon, city_name):
    if index == 0:
        with open(save_path, 'w') as file:
            file.write(f"{index+1}|{dwn_flg}|{ovl_flg}|{wup_pop}|{fll_pop}|{vld_pop}|{region}|{country}|{lat}|{lon}|{city_name}\n")
    else:
        with open(save_path, 'a') as file:
            file.write(f"{index+1}|{dwn_flg}|{ovl_flg}|{wup_pop}|{fll_pop}|{vld_pop}|{region}|{country}|{lat}|{lon}|{city_name}\n")

def open_text(path):
    with open(path, 'r') as files:
        data = files.readlines()
    return data


if __name__ == '__main__':
    main()

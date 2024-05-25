1. Obtain city number
    - dotfiles/h08/global_city/dat/cty_lst_/gpw4/WUP2018_300k_2010.txt
    - 21
2. Load city mask file
    - dat/cty_msk_/gpw4/city_{citynum:08}.gl5
    - dat/vld_cty_/city_{citynum:08}.gl5
3. lndmask
4. Determine geological info like below
    | L | 5184 |
    | XY | “72 72” |
    | LONLAT | “0 6 46 52” |
    | SUF | .pr5 |
5. Create LtoLONLAT conversion file
    - Downloads/H08_20230612/map/pre/prep_basmap.sh
    >>> dat/grd_ara_/grdara.pr5.txt
    >>> dat/l2x_l2y_/l2x.pr5.txt
    >>> dat/l2x_l2y_/l2y.pr5.txt
6. cama
7. rivnxl
    - check if masked{SUFRGN}.CAMA.gl5 is created under /riv_nxl_/ with cama_losangeles.ipynb
    - bash pre/prep_rivnxl_region.sh &
8. mapdat
9. metdat.py
10. ctydat
11. explicitcanal
13. text data
    - map/org/IIASA_SSP/
    - GPC_historical.txt
    - GPC.txt
    - GDP_historical.txt
    - GDP.txt
    - map/dat/nat_cod_/C05_____20000000.txt
    - map/org/MISC_Maps/natwat.txt
14. make_directory.sh
    - no arguments is needed to change
15. move_directory.sh
    - execute after finishing NO.14 
    - change SUFFIX
16. Convert / to = for sharing
    - execute after finishing model run N_C_
    - mkdir {rootdir}/research/H08/regional_model/fileshare/gdrive00000000/cityname
    - convert_filenames.sh
    - line8, 9, 18, 25
17. gunzip the all files
    - tar -cvf savefilename.tar /path/to/savedir/
    - gzip savefilename.tar
1. Obtain city number
    - dotfiles/h08/global_city/dat/cty_lst_/gpw4/WUP2018_300k_2010.txt
2. Load city mask file
    - dat/cty_msk_/gpw4/city_{citynum:08}.gl5
    - dat/vld_cty_/city_{citynum:08}.gl5
3. lndmask
4. Determine geological info like below
    | L | 1728 |
    | XY | “36 48” |
    | LONLAT | “138 141 34 38” |
    | SUF | .tk5 |
5. Create LtoLONLAT conversion file
    - Downloads/H08_20230612/map/pre/prep_basmap.sh
    >>> dat/grd_ara_/grdara.tk5.txt
    >>> dat/l2x_l2y_/l2x.tk5.txt
    >>> dat/l2x_l2y_/l2y.tk5.txt
6. cama
    - create masked{SUFRGN}.CAMA.gl5 under /riv_nxl_/
7. rivnxl
    - pre/prep_rivnxl_region.sh
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
15. move_directory.sh
16. RUN the model
17. Convert / to - for sharing
    - create savedirectory research/H08/regional_model/fileshare/gdrive20240514/riodejaneiro
    - convert_filenames.sh
18. gunzip the all files
    - tar -cvf savefilename.tar /path/to/savedir/
    - gzip savefilename.tar
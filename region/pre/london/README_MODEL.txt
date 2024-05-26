# .sh file
SUF=.ln5
L=1728
XY="48 36"
L2X=../../map/dat/l2x_l2y_/l2x.ln5.txt
L2Y=../../map/dat/l2x_l2y_/l2y.ln5.txt
LONLAT="-3 1 50 53"
ARG="$L $XY $L2X $L2Y $LONLAT"

1. meteorologicla data 
    - met/pre/prep_mean_tk5.sh
        - "LWdown__ Prcp____ PSurf___ Qair____ Rainf___ Snowf___ SWdown__ Tair____ Wind____"
        - ll W5E5____*00.tk5[volume should be less than 36MB(.gl5)]
    - met/pst/calc_koppen_tk5.sh

2. land model
    -  lnd/pre/prep_tk5.sh
    -  lnd/pre/prep_gamtau_tk5.sh
    -  lnd/pre/prep_gwr_tk5.sh
        - if debug mode, error will arises due to no proper debbugging point
    -  lnd/bin/main_tk5.f
        - n0l=1728
        - make all
    -  lnd/bin/main_tk5.sh
        - SUF=.ln5

3. river model
    -  riv/pre/prep_tk5.sh
    -  riv/bin/main_tk5.f
        - n0l=1728
        - make all
    -  riv/bin/main_tk5.sh
        - SUF=.ln5
    -  cpl/pst/list_watbal_tk5.sh
        - RUN="LR__"

4. crop model
    - crp/pre/prep_tk5.sh
    - map/bin/calc_crptyp_tk5.sh
    - map/bin/calc_crpfrc_tk5.sh
    - crp/bin/main_tk5.f
        - n0lall=48*36
        - htstat $ARG sum map/dat/lnd_msk_/lndmsk.CAMA.ln5
        - n0llnd=257
        - (bash sumhtstat.sh)
    - crp/bin/main_tk5.sh(1st crop)
    - crp/pst/calc_crpcal_tk5.sh
    - crp/bin/main_tk5.sh(2nd crop)

5. dam map
    -  riv/pst/calc_mean_tk5.sh
    -  riv/pst/calc_flddro_tk5.sh
    -  map/bin/main_dam_tk5.sh
        - RUN=D_L_ & RUN=D_M_ respectively

6. environmental flow model
    -  riv/pst/calc_envout_tk5.sh

7. intake model
    -  map/pre/prep_map_lcan_tk5.sh
        - MAX=1
    -  prog_map_K14_tk5.f
        - n0l=1728
        - make all
    -  prep_map_K14_tk5.sh
        - produce binary file filled with 0 for exp canal as NONdata

8. desalination model
    -  map/pre/prep_map_IIASA_SSAP_tk5.sh
    -  prog_map_cstlin_tk5.f
        - n0l=1728
        - n0x=48, n0y=36
        - make all
    -  map/prep/prep_map_cstlin_tk5.sh
    -  map/pre/prep_map_despot_tk5.sh

9. couple model 
    -  cpl/pre/prep_tk5.sh
    -  cpl/bin/main_tk5.f
        - n0l=1728
        - make all
    -  cpl/bin/main_tk5.sh(N_C_)
        - adm/Mkinclude
            - switch -debug mode to -03 optimizatoin
        - SUF
        - OPTNNBS=yes
        - DAM=no
    -  cpl/pst/calc_mean_tk5.sh
    -  cpl/bin/main_tk5.sh(LECD)
        - OPTNNBS=no
        - DAM=yes

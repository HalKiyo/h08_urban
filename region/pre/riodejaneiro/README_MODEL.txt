# .sh file
SUF=.ro5
L=4032
XY="84 48"
L2X=../../map/dat/l2x_l2y_/l2x.ro5.txt
L2Y=../../map/dat/l2x_l2y_/l2y.ro5.txt
LONLAT="-47 -40 -24 -20"
ARG="$L $XY $L2X $L2Y $LONLAT"

#.f file
n0l=4032

#Makefile
make all
"""
OBJS = main_tk5
TARGET3 = main_tk5
COMPONENT3 = ${DIRLND}calc_leakyb.o ${DILND}calc_ts_nxt.o \
             ${DIRLIB}read_binary.o ${DILIB}wrte_binary.o \
             ${DIRLIB}read_result.o \
             ${DIRLIB}wrte_bints2.o \
             ${DIRLIB}igetday.o ${DILIB}calc_spinup.o \
             ${DIRLIB}conv_rhtoqa.o ${DILIB}conv_rstors.o \
$(TARGET3) : $(TARGET3).o $(COMPONENT3)
    $(FC) -o $@ $@.o $(COMPONENT3)
"""

1. meteorologicla data 
    - met/pre/prep_mean_tk5.sh
        - "LWdown__ Prcp____ PSurf___ Qair____ Rainf___ Snowf___ SWdown__ Tair____ Wind____"
        - ll W5E5____*00.tk5[volume should be less than 36MB(.gl5)]
    - met/pst/calc_koppen_tk5.sh

2. land model
    -  lnd/pre/prep_tk5.sh
    -  lnd/pre/prep_gamtau_tk5.sh
    -  lnd/pre/prep_gwr_tk5.sh
    -  lnd/bin/main_tk5.f
        - n0l=4032
        - make all
    -  lnd/bin/main_tk5.sh
        - SUF=.ro5

3. river model
    -  riv/bin/main_tk5.f
        - n0l=4032
        - make all
    -  riv/bin/main_tk5.sh
        - SUF=.ro5
    -  cpl/pst/list_watbal_tk5.sh
        - RUN="LR__"

4. crop model
    - crp/bin/main_tk5.f
        - n0lall=84*48
        - htstat $ARG sum map/dat/lnd_msk_/lndmsk.CAMA.ro5
        - n0llnd=770
        - (bash sumhtstat.sh)
    - crp/bin/main_tk5.sh(1st crop)
    - map/bin/calc_crptyp_tk5.sh
    - map/bin/calc_crpfrc_tk5.sh
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
        - n0l=4032
        - make all
    -  map/org/K14/bin2txt.sh
    -  prep_map_K14_tk5.sh

8. desalination model
    -  crp/pre/prep_tk5.sh
    -  map/pre/pre_mapIIASA_SSAP_tk5.sh
    -  prog_map_cstlin_tk5.f
    -  map/prep/prep_map_cstlin_tk5.sh
    -  map/pre/prep_map_despot_tk5.sh

9. couple model 
    -  cpl/bin/main_tk5.sh(N_C_)
        - OPTNNBS=yes
        - DAM=no
    -  cpl/pst/calc_mean_tk5.sh
    -  cpl/bin/main_tk5.sh
        - OPTNNBS=no
        - DAM=yes

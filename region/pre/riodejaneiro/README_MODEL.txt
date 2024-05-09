# .sh file
L=4032
XY="84 48"
L2X=../../map/dat/l2x_l2y_/l2x.ro5.txt
L2Y=../../map/dat/l2x_l2y_/l2y.ro5.txt
LONLAT="-47 -40 -24 -20"
SUF=.ro5

#.f file
n0lall=84*48
n0llnd=382
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

4. crop model
    - crp/bin/main_tk5.sh(1st crop)
    - map/bin/calc_crptyp_tk5.sh
    - map/bin/calc_crpfrc_tk5.sh
    - crp/bin/main_tk5.sh(2nd crop)

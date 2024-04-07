# !/bin/bash
#
# load cty_lst_
# conversion from lon lat coordinate of city center
# to x y coordinate in global 5min resolution map
# file is saved to cty_cnt_
#
############################################################
# Geography
############################################################
L2X=/mnt/c/Users/tsimk/Downloads/H08_20230612/map/dat/l2x_l2y_/l2x.gl5.txt
L2Y=/mnt/c/Users/tsimk/Downloads/H08_20230612/map/dat/l2x_l2y_/l2y.gl5.txt
ARG="9331200 4320 2160 $L2X $L2Y -180 180 -90 90"
SUF=.gl5

############################################################
# Input
############################################################
CTYLST=/home/kajiyama/H08/H08_20230612/map/dat/cty_lst_/city_list03.txt
#
############################################################
# Output
############################################################
DIROUT=../../map/dat/cty_cnt_/
#
############################################################
# Job
############################################################
for ID in `seq 1 900`; do
  ID8=`echo $ID | awk '{printf("%8.8d",$1)}'`
  LON=`awk '($1=="'$ID'"){print $3}' $CTYLST`       # need check
  LAT=`awk '($1=="'$ID'"){print $4}' $CTYLST`       # need check
  NAME=`awk '($1=="'$ID'"){print $6}' $CTYLST`      # need check

  echo ID: $ID8

  OUT=${DIROUT}city_${ID8}${SUF}
  htcreate 9331200 0 ${OUT}
  htedit $ARG lonlat ${OUT} 1 $LON $LAT

done


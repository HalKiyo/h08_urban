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
POP=gpw4
SUF=.30s
L2X=/mnt/c/Users/tsimk/Downloads/H08_20230612/map/dat/l2x_l2y_/l2x.${SUF}.txt
L2Y=/mnt/c/Users/tsimk/Downloads/H08_20230612/map/dat/l2x_l2y_/l2y.${SUF}.txt
ARG="933120000 43200 21600 $L2X $L2Y -180 180 -90 90"

############################################################
# Input
############################################################
CTYLST=/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/dat/cty_lst_/${POP}/WUP2018_300k_2010.txt
#
############################################################
# Output
############################################################
DIROUT=/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/dat/cty_cnt_/${POP}/
#
############################################################
# Job
############################################################
for ID in `seq 1 1860`; do
  ID8=`echo $ID | awk '{printf("%8.8d",$1)}'`
  LAT=`awk '($1=="'$ID'"){print $2}' $CTYLST`       # need check
  LON=`awk '($1=="'$ID'"){print $3}' $CTYLST`       # need check
  NAME=`awk '($1=="'$ID'"){print $5}' $CTYLST`      # need check

  echo ID: $ID8 $LAT $LON $NAME

  OUT=${DIROUT}city_${ID8}${SUF}
  htcreate 9331200 0 ${OUT}
  htedit $ARG lonlat ${OUT} 1 $LON $LAT

done


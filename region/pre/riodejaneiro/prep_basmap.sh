#!/bin/sh
############################################################
#to   prepare domain
#by   2010/03/31, hanasaki, NIES
############################################################
# Geographical setting (Edit here if you change spatial domain/resolution)
############################################################
#0.5deg x 0.5deg of globe (.hlf)
#L=259200
#XY="720 360"
#LONLAT="-180 180 -90 90"
#SUF=.hlf

#1deg x 1deg of globe (.one)
#L=64800
#XY="360 180"
#LONLAT="-180 180 -90 90"
#SUF=.one

#5min x 5min of globe (.gl5)
#L=9331200
#XY="4320 2160"
#LONLAT="-180 180 -90 90"
#SUF=.gl5

#5min x 5min of tokyo (.tk5)
#L=1728
#XY="36 48"
#LONLAT="138 141 34 38"
#SUF=.tk5

#5min x 5min of rio (.ro5)
L=4032
XY="84 48"
LONLAT="-47 -40 -24 -20"
SUF=.ro5

#30sec x 30sec of global (.30s)
#L=933120000
#XY="43200 21600"
#LONLAT="-180 180 -90 90"
#SUF=.30s

############################################################
# Output (Do not edit here unless you are an expert)
############################################################
DIRGRDARA=../dat/grd_ara_
DIRL2XL2Y=../dat/l2x_l2y_
#
GRDARA=$DIRGRDARA/grdara${SUF}
L2X=${DIRL2XL2Y}/l2x${SUF}.txt
L2Y=${DIRL2XL2Y}/l2y${SUF}.txt
############################################################
# Job (prepare output directory)
############################################################
if [ !  -d ${DIRL2XL2Y} ]; then   mkdir -p ${DIRL2XL2Y}; fi
if [ !  -d ${DIRGRDARA} ]; then   mkdir -p ${DIRGRDARA}; fi
############################################################
# Job (make files)
############################################################
htl2xl2y $L $XY $L2X $L2Y
prog_grdara $L $XY $L2X $L2Y $LONLAT $GRDARA

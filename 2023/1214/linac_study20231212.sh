#!/bin/sh

exec=Linac20231212.py
conf=linac20231212_5.ini

ixf=1
while [ $ixf -le 5 ]
do
    date=`date +'%Y%m%d_%H:%M'`
    #echo $exec $conf
    python3.11 $exec $conf |& tee log.$date
    ixf=$(( ixf + 1 ))
done

#!/bin/sh


#exec=Linac20231214.py
#conf=linac20231214_1.ini
#
#ixf=1
#while [ $ixf -le 5 ]
#do
#    date=`date +'%Y%m%d_%H:%M'`
#    #echo $exec $conf
#    python3.11 $exec $conf |& tee log.$date
#    ixf=$(( ixf + 1 ))
#done


exec=Linac20231214.py
conf=linac20231214_2.ini

ixf=1
while [ $ixf -le 3 ]
do
    date=`date +'%Y%m%d_%H:%M'`
    #echo $exec $conf
    python3.11 $exec $conf |& tee log.$date
    ixf=$(( ixf + 1 ))
done

exec=Linac20231214.py
conf=linac20231214_3.ini

ixf=1
while [ $ixf -le 3 ]
do
    date=`date +'%Y%m%d_%H:%M'`
    #echo $exec $conf
    python3.11 $exec $conf |& tee log.$date
    ixf=$(( ixf + 1 ))
done

exec=Linac20231214.py
conf=linac20231214_4.ini

ixf=1
while [ $ixf -le 3 ]
do
    date=`date +'%Y%m%d_%H:%M'`
    #echo $exec $conf
    python3.11 $exec $conf |& tee log.$date
    ixf=$(( ixf + 1 ))
done

exec=Linac20231214.py
conf=linac20231214_5.ini

ixf=1
while [ $ixf -le 3 ]
do
    date=`date +'%Y%m%d_%H:%M'`
    #echo $exec $conf
    python3.11 $exec $conf |& tee log.$date
    ixf=$(( ixf + 1 ))
done

exec=Linac20231214.py
conf=linac20231214_6.ini

ixf=1
while [ $ixf -le 3 ]
do
    date=`date +'%Y%m%d_%H:%M'`
    #echo $exec $conf
    python3.11 $exec $conf |& tee log.$date
    ixf=$(( ixf + 1 ))
done

exec=Linac20231214.py
conf=linac20231214_7.ini

ixf=1
while [ $ixf -le 5 ]
do
    date=`date +'%Y%m%d_%H:%M'`
    #echo $exec $conf
    python3.11 $exec $conf |& tee log.$date
    ixf=$(( ixf + 1 ))
done

exec=Linac20231214.py
conf=linac20231214_8.ini

ixf=1
while [ $ixf -le 5 ]
do
    date=`date +'%Y%m%d_%H:%M'`
    #echo $exec $conf
    python3.11 $exec $conf |& tee log.$date
    ixf=$(( ixf + 1 ))
done

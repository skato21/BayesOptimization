#!../../bin/linux-x86_64/libera

## You may have to change libera to something else
## everywhere it appears in this file

< envPaths

cd "${TOP}"

## Register all support components
dbLoadDatabase "dbd/libera.dbd"
libera_registerRecordDeviceDriver pdbbase

## Load record instances
dbLoadRecords "db/dbStriplineHER.db", "user=mitsukaHost"

## Set this to see messages from mySub
#var mySubDebug 1

## Run this to trace the stages of iocInit
#traceIocInit

cd "${TOP}/iocBoot/${IOC}"
iocInit

## Start any sequence programs
seq StriplineHER, "user=mitsukaHost"

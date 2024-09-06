TOP=../..

include $(TOP)/configure/CONFIG

liberaSupport_LIBS += $(EPICS_BASE_IOC_LIBS)

PROD_IOC = libera

DBD += libera.dbd

# libera_registerRecordDeviceDriver.cpp derives from libera.dbd
libera_SRCS += libera_registerRecordDeviceDriver.cpp
libera_LIBS += $(EPICS_BASE_IOC_LIBS)

# Build the main IOC entry point where needed
libera_SRCS_DEFAULT += liberaMain.cpp
libera_SRCS_vxWorks += -nil-

# Build StriplineLER into liberaSupport
libera_SNCFLAGS += +r
libera_SRCS  += bobyqa.cpp
libera_SRCS  += fitfunc.cpp
libera_DBD   += StriplineLER.dbd
libera_SRCS  += StriplineLER.stt
libera_DBD   += StriplineHER.dbd
libera_SRCS  += StriplineHER.stt
libera_DBD   += base.dbd
libera_LIBS  += seq pv

gsl_DIR      += /usr/lib/x86_64-linux-gnu
PROD_LIBS    += gsl
gslcblas_DIR += /usr/lib/x86_64-linux-gnu
PROD_LIBS    += gslcblas

# Finally link IOC to the EPICS Base libraries
libera_LIBS += $(EPICS_BASE_IOC_LIBS)

include $(TOP)/configure/RULES
#----------------------------------------
#  ADD EXTRA GNUMAKE RULES BELOW HERE

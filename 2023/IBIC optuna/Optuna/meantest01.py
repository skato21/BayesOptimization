import optuna
import numpy as np
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

from bayeso_benchmarks import Hartmann6D, Colville
from botorch.settings import validate_input_scaling

from epics import PV, caget, caput

import logging
import sys
import threading
import time, datetime
import configparser
import re

# --- Load configuration ---
args = sys.argv
config_ini = configparser.ConfigParser()
config_ini.read(args[1])

ny = int(config_ini.get('PV', 'ny'))

ypv = []
yalias = []
y = {}
for i in range(ny):
    pv = PV(config_ini.get('PV_Y{0}'.format(i), 'name'))
    ypv.append(pv)
    yalias.append(config_ini.get('PV_Y{0}'.format(i), 'alias'))
    y[yalias[i]] = 0
    
pv_E = PV("TEST:E")
pv_X = PV("TEST:Xpos")
pv_Y = PV("TEST:Ypos")
pv_Q = PV("TEST:Q")

pv_E_mean = PV("TEST:E_mean")
pv_X_mean = PV("TEST:Xpos_mean")
pv_Y_mean = PV("TEST:Ypos_mean")
pv_Q_mean = PV("TEST:Q_mean")

l_dispesion_value = []
l_xSqAve_value = []
l_ySqAve_value = []
l_QAve_value = []

def measure():
    while True:
        global stop_threads

        fin = open('switch', 'r')
        data = fin.read()
        match = re.match(data, 'exit')
        fin.close()

        if stop_threads or match is not None:
            print('break now')
            break

        for i, pv in enumerate(ypv):
            y[yalias[i]] = pv.get()

        dispesion_value \
            = y['EX_R0_62']**2 + y['EY_R0_62']**2 + y['EX_R0_63']**2 + y['EY_R0_63']**2 \
            + y['EX_C1_4']**2 + y['EX_C2_4']**2 + y['EX_C3_4']**2 + y['EX_C4_4']**2 \
            + y['EX_C5_4']**2 + y['EX_C6_4']**2 + y['EX_C7_4']**2 + y['EX_C8_4']**2 \
            + y['EY_C1_4']**2 + y['EY_C2_4']**2 + y['EY_C3_4']**2 + y['EY_C4_4']**2 \
            + y['EY_C5_4']**2 + y['EY_C6_4']**2 + y['EY_C7_4']**2 + y['EY_C8_4']**2 \
            + y['EX_11_4']**2 + y['EX_12_4']**2 + y['EX_13_5']**2 + y['EX_15_T']**2 \
            + y['EY_11_4']**2 + y['EY_12_4']**2 + y['EY_13_5']**2 + y['EY_15_T']**2

        xSqAve_value \
            = y['x_R0_62']**2 + y['x_R0_63']**2 \
            + y['x_C1_4']**2 + y['x_C2_4']**2 + y['x_C3_4']**2 + y['x_C4_4']**2 \
            + y['x_C5_4']**2 + y['x_C6_4']**2 + y['x_C7_4']**2 + y['x_C8_4']**2 \
            + y['x_11_4']**2 + y['x_12_4']**2 + y['x_13_5']**2 + y['x_15_T']**2
        xSqAve_value /= 14

        ySqAve_value \
            = y['y_R0_62']**2 + y['y_R0_63']**2 \
            + y['y_C1_4']**2 + y['y_C2_4']**2 + y['y_C3_4']**2 + y['y_C4_4']**2 \
            + y['y_C5_4']**2 + y['y_C6_4']**2 + y['y_C7_4']**2 + y['y_C8_4']**2 \
            + y['y_11_4']**2 + y['y_12_4']**2 + y['y_13_5']**2 + y['y_15_T']**2
        ySqAve_value /= 14

        QAve_value \
            = y['Q_R0_62'] + y['Q_R0_63'] \
            + y['Q_C1_4'] + y['Q_C2_4'] + y['Q_C3_4'] + y['Q_C4_4'] \
            + y['Q_C5_4'] + y['Q_C6_4'] + y['Q_C7_4'] + y['Q_C8_4'] \
            + y['Q_11_4'] + y['Q_12_4'] + y['Q_13_5'] + y['Q_15_T']
        QAve_value /= 14
		
        pv_E.put(dispesion_value)
        pv_X.put(xSqAve_value)
        pv_Y.put(ySqAve_value)
        pv_Q.put(QAve_value)

        if len(l_dispesion_value) > 2:
            l_dispesion_value.pop(0)
        l_dispesion_value.append(dispesion_value)
        dispesion_value = np.mean(l_dispesion_value)

        if len(l_xSqAve_value) > 2:
            l_xSqAve_value.pop(0)
        l_xSqAve_value.append(xSqAve_value)
        xSqAve_value = np.mean(l_xSqAve_value)

        if len(l_ySqAve_value) > 2:
            l_ySqAve_value.pop(0)
        l_ySqAve_value.append(ySqAve_value)
        ySqAve_value = np.mean(l_ySqAve_value)

        if len(l_QAve_value) > 2:
            l_QAve_value.pop(0)
        l_QAve_value.append(QAve_value)
        QAve_value = np.mean(l_QAve_value)

        pv_E_mean.put(dispesion_value)
        pv_X_mean.put(xSqAve_value)
        pv_Y_mean.put(ySqAve_value)
        pv_Q_mean.put(QAve_value)

        time.sleep(10)

stop_threads = False
t_measure = threading.Thread(target = measure)
t_measure.start()

# time.sleep(60)

# stop_threads = True
# t_measure.join()

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

def Haltmann6D_EPICS_fun(inputs):
    caput ('TEST:X0', inputs[0])
    caput ('TEST:X1', inputs[1])
    caput ('TEST:X2', inputs[2])
    caput ('TEST:X3', inputs[3])
    caput ('TEST:X4', inputs[4])
    caput ('TEST:X5', inputs[5])
    time.sleep (0.1)
    return caget('TEST:Y')

print(Haltmann6D_EPICS_fun([0.5,0.5,0.5,0.5,0.5,0.5]))

for i in range (1,1):

    print(i)

study = optuna.create_study()


n_trials = 50 



for n in range(50):
    trial = study.ask()

    #最適化の具体的な中身
    
    study.tell(trial, Test_fun(X))




def objective(trial):
    
    #最適化の具体的な中身
    
    return Test_fun(X)

study.optimize(objective, n_trials=50)


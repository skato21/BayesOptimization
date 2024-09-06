import optuna
import numpy as np
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

from botorch.settings import validate_input_scaling

from epics import PV, caget, caput

import logging
import sys
import threading
import time
import configparser

# --- Load configuration ---
args = sys.argv
config_ini = configparser.ConfigParser()
config_ini.read(args[1])

nxd = int(config_ini.get('PV', 'nxd'))
ny = int(config_ini.get('PV', 'ny'))
evalsleep = float(config_ini.get('PV', 'evalsleep'))

xpv = []
xrange = []
for i in range(nxd):
    pv = PV(config_ini.get('PV_XD{0}'.format(i), 'name'))
    xpv.append(pv)
    rmin = float(config_ini.get('PV_XD{0}'.format(i), 'rmin'))
    rmax = float(config_ini.get('PV_XD{0}'.format(i), 'rmax'))
    xrange.append(list([rmin, rmax]))

ypv = []
yalias = []
y = {}
for i in range(ny):
    pv = PV(config_ini.get('PV_Y{0}'.format(i), 'name'))
    ypv.append(pv)

# --- Objective function ---
def objective(trial):
    for i, pv in enumerate(xpv):
        x = trial.suggest_float('x{0}'.format(i), xrange[i][0], xrange[i][1])
        pv.put(x)

    time.sleep(evalsleep/1000.)

    ysum = 0.
    for i, pv in enumerate(ypv):
        ysum += pv.get()
    
    return ysum, xpv[0].get()

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "EXample-study"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)

# --- Create black-box optimization ---
study = optuna.create_study(
    # study_name=study_name,
    sampler=optuna.integration.BoTorchSampler(),
    # sampler=optuna.samplers.TPESampler(),
    #sampler=optuna.samplers.CmaEsSampler(),
    storage=storage_name,
    directions=["minimize", "maximize"]
)

study.optimize(objective, n_trials=100)

# --- Best params & trials ---
for trial in study.best_trials:
    print(f"- [{trial.number}] params={trial.params}, values={trial.values}")

# --- Pareto front plot ---
optuna.visualization.plot_pareto_front(
    study,
    include_dominated_trials=True
    # include_dominated_trials=False
).show()

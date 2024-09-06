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

# --- Load configuration ---
args = sys.argv
config_ini = configparser.ConfigParser()
config_ini.read(args[1])

nxd = int(config_ini.get('PV', 'nxd'))
nxr = int(config_ini.get('PV', 'nxr'))
ny = int(config_ini.get('PV', 'ny'))
evalsleep = float(config_ini.get('PV', 'evalsleep'))

xpv = []
xrange = []
for i in range(nxr):
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
#    yalias.append(config_ini.get('PV_Y{0}'.format(i), 'alias'))
#    y[yalias[i]] = 0

l_dispesion_value = []
l_xSqAve_value = []
l_ySqAve_value = []
l_QAve_value = []

def Haltmann6D_EPICS_fun(inputs):
    caput ('TEST:X0', inputs[0])
    caput ('TEST:X1', inputs[1])
    caput ('TEST:X2', inputs[2])
    caput ('TEST:X3', inputs[3])
    caput ('TEST:X4', inputs[4])
    caput ('TEST:X5', inputs[5])
    time.sleep (0.1)
    return caget('TEST:Y')

#print(xrange[3][0])

# --- Objective function ---
def objective(trial):
    x0 = trial.suggest_float('TEST:X0', xrange[0][0], xrange[0][1])
    x1 = trial.suggest_float('TEST:X1', xrange[1][0], xrange[1][1])
    x2 = trial.suggest_float('TEST:X2', xrange[2][0], xrange[2][1])
    x3 = trial.suggest_float('TEST:X3', xrange[3][0], xrange[3][1])
    x4 = trial.suggest_float('TEST:X4', xrange[4][0], xrange[4][1])
    x5 = trial.suggest_float('TEST:X5', xrange[5][0], xrange[5][1])

    '''
    for i, pv in enumerate(xpv):
        x = trial.suggest_float('x{0}'.format(i), xrange[i][0], xrange[i][1])
        pv.put(x)
    '''

    time.sleep(evalsleep/1000.)

    return Haltmann6D_EPICS_fun([x0,x1,x2,x3,x4,x5])

vysum = []
class StopWhenTrialKeepBeingUnupdated:
    def __init__(self, distance: int, threshold: float):
        self.distance = distance
        self.threshold = threshold

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.Trial) -> None:
        vysum.append(study.best_value)

        if trial.number > self.distance:
            if (abs(vysum[trial.number-self.distance] - study.best_value)/study.best_value) < self.threshold:
                print(trial.number, vysum[trial.number - self.distance], study.best_value)
                study.stop()

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

#source_study = optuna.load_study(
#    study_name="2023_05_28_14_44_10",
#    storage="sqlite:///optuna_dispersion02.db"
#)

now = datetime.datetime.now()
filename = now.strftime('%Y_%m_%d_%H_%M_%S')

study = optuna.create_study(
    #sampler=optuna.samplers.TPESampler(),
    # sampler=optuna.samplers.CmaEsSampler(source_trials=source_study.trials),
    # sampler=optuna.samplers.CmaEsSampler(),
    direction = 'minimize',
    sampler=optuna.integration.BoTorchSampler(),
    study_name="{}".format(filename),
    storage="sqlite:///optuna_Haltmann6D.db"
)

#for trial in sorted(source_study.trials, key=lambda t: t.value)[:10]:
#    study.enqueue_trial(trial.params)

#stop_threads = False
#t_measure = threading.Thread(target = measure)
#t_measure.start()

study_stop_cb = StopWhenTrialKeepBeingUnupdated(100, 1e-4)
study.optimize(objective, n_trials=50, callbacks=[study_stop_cb])

#stop_threads = True
#t_measure.join()

# --- Best results ---
print(f"- Best objective value: {study.best_value}")

best_params = study.best_params
for i, val in enumerate(best_params):
    xpv[i].put(best_params[val])
    print(f"- Best {i} parameter: {best_params[val]}")

# optuna.visualization.plot_param_importances(study).show()
# plot_optimization_history(study).show()
# plot_parallel_coordinate(study).show()
# plot_contour(study).show()

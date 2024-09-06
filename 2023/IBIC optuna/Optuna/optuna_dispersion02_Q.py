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
    yalias.append(config_ini.get('PV_Y{0}'.format(i), 'alias'))
    y[yalias[i]] = 0

l_dispesion_value = []
l_xSqAve_value = []
l_ySqAve_value = []
l_QAve_value = []

def measure():
    while True:
        global stop_threads, dispesion_value, xSqAve_value, ySqAve_value, QAve_value

        if stop_threads:
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

        time.sleep(3)

# --- Objective function ---
xpre = list(range(nxd))
cons = list(range(nxd))
def objective(trial):
    for i, pv in enumerate(xpv):
        x = trial.suggest_float('x{0}'.format(i), xrange[i][0], xrange[i][1])
        pv.put(x)

        cons[i] = abs(xpre[i] - x) - 1.0
        xpre[i] = x

    time.sleep(evalsleep/1000.)

    trial.set_user_attr("constraints", (cons[0], cons[1], cons[2], cons[3], cons[4], cons[5]))

    return QAve_value

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

# source_study = optuna.load_study(
#     study_name="2023_06_12_11_02_44",
#     storage="sqlite:///optuna_dispersion02_Q1.db"
# )

now = datetime.datetime.now()
filename = now.strftime('%Y_%m_%d_%H_%M_%S')

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.integration.BoTorchSampler(constraints_func=lambda trial: trial.user_attrs["constraints"]),
    # sampler=optuna.samplers.TPESampler(),
    # sampler=optuna.samplers.CmaEsSampler(),
    # sampler=optuna.samplers.CmaEsSampler(source_trials=source_study.trials),
    study_name="{}".format(filename),
    storage="sqlite:///optuna_dispersion02_Q1.db"
)

# for trial in sorted(source_study.trials, key=lambda t: t.value)[90:]:
#     study.enqueue_trial(trial.params)

stop_threads = False
t_measure = threading.Thread(target = measure)
t_measure.start()

study_stop_cb = StopWhenTrialKeepBeingUnupdated(500, 1e-4)
study.optimize(objective, n_trials=500, callbacks=[study_stop_cb])

stop_threads = True
t_measure.join()

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

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
    yalias.append(config_ini.get('PV_Y{0}'.format(i), 'alias'))
    y[yalias[i]] = 0

obj_fun = Colville()
def objective(trial):
    x0 = trial.suggest_float('x0', 0, 2)
    x1 = trial.suggest_float('x1', 0, 2)
    x2 = trial.suggest_float('x2', 0, 2)
    x3 = trial.suggest_float('x3', 0, 2)
    arr = np.array([x0, x1, x2, x3])

    # for i, pv in enumerate(xpv):
    #     x = trial.suggest_float('x{0}'.format(i), xrange[i][0], xrange[i][1])
    #     pv.put(x)

    time.sleep(evalsleep/1000.)

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

    print('DEBUG ',dispesion_value, QAve_value, xSqAve_value, ySqAve_value)

    # return dispesion_value*(1/QAve_value + xSqAve_value + ySqAve_value)

    return obj_fun.function(arr)

# optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
# study_name = "y['EXample-study"  # Unique identifier of the study.
# storage_name = "sqlite:///{}.db".format(study_name)

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

study = optuna.create_study(
    # study_name=study_name,
    sampler=optuna.integration.BoTorchSampler(),
    # sampler=optuna.samplers.TPESampler(),
    #sampler=optuna.samplers.CmaEsSampler(),
    # storage=storage_name
)

study_stop_cb = StopWhenTrialKeepBeingUnupdated(30, 1e-4)
study.optimize(objective, n_trials=300, callbacks=[study_stop_cb])

best_params = study.best_params
print(best_params)
plt0 = plot_optimization_history(study)
plt1 = plot_parallel_coordinate(study)
plt2 = plot_contour(study)

plt0.show()
plt1.show()
plt2.show()

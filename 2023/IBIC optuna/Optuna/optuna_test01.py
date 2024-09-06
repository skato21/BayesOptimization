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
import time, datetime
import configparser

#--- Load configuration ---
args = sys.argv
config_ini = configparser.ConfigParser()
config_ini.read(args[1])

evalsleep = float(config_ini.get('PV', 'evalsleep'))

# xpv = []
# xrange = []
# nxd = int(config_ini.get('PV', 'nxd'))
# for i in range(nxd):
#     pv = PV(config_ini.get('PV_XD{0}'.format(i), 'name'))
#     xpv.append(pv)
#     rmin = float(config_ini.get('PV_XD{0}'.format(i), 'rmin'))
#     rmax = float(config_ini.get('PV_XD{0}'.format(i), 'rmax'))
#     xrange.append(list([rmin, rmax]))

# ypv = []
# yweight = []
# ny  = int(config_ini.get('PV', 'ny'))
# for i in range(ny):
#     pv = PV(config_ini.get('PV_Y{0}'.format(i), 'name'))
#     ypv.append(pv)
#     weight = float(config_ini.get('PV_Y{0}'.format(i), 'weight'))
#     yweight.append(weight)

# def objective(trial):
#     for i, pv in enumerate(xpv):
#         x = trial.suggest_float('x{0}'.format(i), xrange[i][0], xrange[i][1])
#         pv.put(x)

#     time.sleep(evalsleep/1000.)

#     ysum = 0.;
#     for i, pv in enumerate(ypv):
#         ysum += pv.get() * yweight[i]
    
#     return ysum

xpre = list(range(4))
cons = list(range(4))
obj_fun = Colville()
def objective(trial):
    x0 = trial.suggest_float('x0', 0, 2)
    x1 = trial.suggest_float('x1', 0, 2)
    x2 = trial.suggest_float('x2', 0, 2)
    x3 = trial.suggest_float('x3', 0, 2)
    arr = np.array([x0, x1, x2, x3])

    cons[0] = abs(xpre[0] - x0) - 1.0
    cons[1] = abs(xpre[1] - x1) - 1.0
    cons[2] = abs(xpre[2] - x2) - 1.0
    cons[3] = abs(xpre[3] - x3) - 1.0

    xpre[0] = x0
    xpre[1] = x1
    xpre[2] = x2
    xpre[3] = x3

    trial.set_user_attr("constraints", (cons[0], cons[1], cons[2], cons[3]))

    return obj_fun.function(arr)

vysum = []
class StopWhenTrialKeepBeingUnupdated:
    def __init__(self, distance: int, threshold: float):
        self.distance = distance
        self.threshold = threshold

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.Trial) -> None:
        vysum.append(study.best_value)

        if trial.number > self.distance:
            if abs((vysum[trial.number-self.distance] - study.best_value)/study.best_value) < self.threshold:
                # print(trial.number, vysum[trial.number - self.distance], study.best_value)
                study.stop()

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# source_study = optuna.load_study(
#     study_name="2023_05_27_10_46_03",
#     storage="sqlite:///optuna_test01.db"
# )

now = datetime.datetime.now()
filename = now.strftime('%Y_%m_%d_%H_%M_%S')

study = optuna.create_study(
    sampler=optuna.integration.BoTorchSampler(constraints_func=lambda trial: trial.user_attrs["constraints"]),
    # sampler=optuna.samplers.TPESampler(),
    # sampler=optuna.samplers.CmaEsSampler(source_trials=source_study.trials),
    study_name="{}".format(filename), # Unique identifier of the study.
    storage="sqlite:///optuna_test01.db"
)

# for trial in sorted(source_study.trials, key=lambda t: t.value)[:3]:
#     study.enqueue_trial(trial.params)

study_stop_cb = StopWhenTrialKeepBeingUnupdated(300, 1e-4)
study.optimize(objective, n_trials=300, callbacks=[study_stop_cb])

print(f"- Best objective value: {study.best_value}")

best_params = study.best_params
for i, val in enumerate(best_params):
    # xpv[i].put(best_params[val])
    print(f"- Best {i} parameter: {best_params[val]}")

optuna.visualization.plot_param_importances(study).show()
plot_optimization_history(study).show()
plot_parallel_coordinate(study).show()
plot_contour(study).show()

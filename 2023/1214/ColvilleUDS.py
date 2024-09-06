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
xinit = []
xpvname = []
xstep = []

for i in range(nxr):
    pv = PV(config_ini.get('PV_XD{0}'.format(i), 'name'))
    xpv.append(pv)
    rmin = float(config_ini.get('PV_XD{0}'.format(i), 'rmin'))
    rmax = float(config_ini.get('PV_XD{0}'.format(i), 'rmax'))
    xrange.append(list([rmin, rmax]))
    step = float(config_ini.get('PV_XD{0}'.format(i), 'step'))
    xstep.append(step)
    init = float(config_ini.get('PV_XD{0}'.format(i), 'init'))
    xinit.append(init)
    pvname = config_ini.get('PV_XD{0}'.format(i), 'name')
    xpvname.append(pvname)

ypv = []
ypvname = []
yalias = []

for i in range(ny):
    pv = PV(config_ini.get('PV_Y{0}'.format(i), 'name'))
    ypv.append(pv)
    pvname = config_ini.get('PV_Y{0}'.format(i), 'name')
    ypvname.append(pvname)
#    yalias.append(config_ini.get('PV_Y{0}'.format(i), 'alias'))
#    y[yalias[i]] = 0

l_dispesion_value = []
l_xSqAve_value = []
l_ySqAve_value = []
l_QAve_value = []

def Test_fun(inputs):
    for i in range (len(xpv)):
        caput (xpvname[i], inputs[i])
    time.sleep (0.1)
    return caget(ypvname[0])

class StopWhenTrialKeepBeingUnupdated:
    def __init__(self, distance: int, threshold: float):
        self.distance = distance
        self.threshold = threshold

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.Trial) -> None:
        vysum.append(study.best_value)

        if trial.number > self.distance:
            if (
                abs(vysum[trial.number - self.distance] - study.best_value)
                / study.best_value
            ) < self.threshold:
                print(
                    trial.number, vysum[trial.number - self.distance], study.best_value
                )
                study.stop()

 #Xoldの初期値はinitilizeした値を使う
dx = [0 for i in range (len(xpv))]
xold = [0 for i in range (len(xpv))]
nstep = [0 for i in range (len(xpv))]
dstep = [0 for i in range (len(xpv))]
X = [0 for i in range (len(xpv))] #空リストだとなぜか怒られる

# --- Objective function ---
def objective(trial):
    
    
    for i in range (len(xpv)):
        X[i] = trial.suggest_float(xpvname[i], xrange[i][0], xrange[i][1])
        
    '''for i in range(len(steplist)):
        dx = X[i] - Xold[i]             #i回目から(i-1)回目の差分
        nstep = int(abs(dx) / xstep[i]) + 1    #差分から、何点を間に挟むか
        dstep = dx / nstep                        #差分を間に挟む点で割った数
        
        for j in range (1, nstep + 1):
            caput(xpvname[j],dstep * nstep + Xold[j])
            time.sleep(evalsleep / 1000.0)
    '''#やや非効率なやり方
    
    Xold = [0,0,0,0]
    for i in range(len(xpv)):
        dx[i] = X[i] - Xold[i]                          #i回目から(i-1)回目の差分
        nstep[i] = int(abs(dx[i]) / xstep[i]) + 1    #差分から、何点を間に挟むか
        dstep[i] = dx[i] / nstep[i]                     #差分を間に挟む点で割った数
        
    for j in range (1,max(nstep)+1):
        for k in range(len(xpv)):
            if j <= nstep[k]:
                caput(xpvname[k],dstep[k] * nstep[k] + Xold[k])
                print("{0}トライアルのx{1}を{2}/{3}分割目".format(number,k,j,nstep[k]))
        time.sleep(evalsleep / 1000.0)

    Xold = X
    return Test_fun(X)

vysum = []

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# source_study = optuna.load_study(
#    study_name="2023_05_28_14_44_10",
#    storage="sqlite:///optuna_dispersion02.db"
# )

now = datetime.datetime.now()
filename = now.strftime("%Y_%m_%d_%H_%M_%S")


# ここからBoTorchSamplerをパクってclassを作る


import Hitohudebayes

study = optuna.create_study(
    # sampler=optuna.samplers.TPESampler(),
    # sampler=optuna.samplers.CmaEsSampler(source_trials=source_study.trials),
    # sampler=optuna.samplers.CmaEsSampler(),
    direction="minimize",
    # sampler=optuna.integration.BoTorchSampler(),
    sampler=Hitohudebayes.HitohudebayesSampler(),
    study_name="{}".format(filename),
    storage="sqlite:///optuna_ColvilleUDS.db",
)

# for trial in sorted(source_study.trials, key=lambda t: t.value)[:10]:
#    study.enqueue_trial(trial.params)

# stop_threads = False
# t_measure = threading.Thread(target = measure)
# t_measure.start()

study_stop_cb = StopWhenTrialKeepBeingUnupdated(100, 1e-4)
study.optimize(objective, n_trials=100, callbacks=[study_stop_cb])

# stop_threads = True
# t_measure.join()

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

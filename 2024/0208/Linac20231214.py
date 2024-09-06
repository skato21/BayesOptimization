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


import csv
import datetime
import os

import logging
import sys
import threading
import time, datetime
import configparser
import TheSummer.Hitohudebayes as Hitohudebayes
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

# --- Load configuration ---
args = sys.argv
config_ini = configparser.ConfigParser()
config_ini.read(args[1])
nxd = int(config_ini.get("PV", "nxd"))
nxr = int(config_ini.get("PV", "nxr"))
ny = int(config_ini.get("PV", "ny"))
evalsleep = float(config_ini.get("PV", "evalsleep"))

xpv = []
xrange = []
xinit = {} #optunaに引き渡す都合上、辞書型にしている
xpvname = []
xstep = []
xweight= []

for i in range(nxr):
    pv = PV(config_ini.get("PV_XD{0}".format(i), "name"))
    xpv.append(pv)
    rmin = float(config_ini.get("PV_XD{0}".format(i), "rmin"))
    rmax = float(config_ini.get("PV_XD{0}".format(i), "rmax"))
    xrange.append(list([rmin, rmax]))
    step = float(config_ini.get("PV_XD{0}".format(i), "step"))
    xstep.append(step)
    init = float(config_ini.get("PV_XD{0}".format(i), "init"))
    xinit[config_ini.get("PV_XD{0}".format(i), "name")] = init
    weight = float(config_ini.get("PV_XD{0}".format(i), "weight"))
    xweight.append(weight)
    pvname = config_ini.get("PV_XD{0}".format(i), "name")
    xpvname.append(pvname)

ypv = []
ypvname = []
yalias = []

for i in range(ny):
    pv = PV(config_ini.get("PV_Y{0}".format(i), "name"))
    ypv.append(pv)
    pvname = config_ini.get("PV_Y{0}".format(i), "name")
    ypvname.append(pvname)
#    yalias.append(config_ini.get('PV_Y{0}'.format(i), 'alias'))
#    y[yalias[i]] = 0


number_of_measurements = int(config_ini.get("PV", "number_of_measurements"))
Func_average = []


def measurement(inputs):
    # 各パラメータを複数回測定する
    for i in range(nxr):
        caput(xpvname[i], inputs[i])
    time.sleep(1)    
    for j in range(number_of_measurements):  # 評価値をnumber_of_measurements回測定して平均を取る
        time.sleep(evalsleep / 1000.0)
        Func_average.append(caget(ypvname[0]))
    Func_value = np.mean(Func_average)
    Func_average.clear()  # リストの中身を空にする。

    # 区間平均を取る
    #    for i in range(len(xpv)):
    #        caput(xpvname[i], inputs[i])
    #    time.sleep(evalsleep / 1000.0)
    #    if len(Func_average) > 2:
    #            Func_average.pop(0)
    #    Func_average.append(caget(ypvname[0]))
    #    Func_value = np.mean(Func_average)

    return Func_value


now = datetime.datetime.now()
filename = now.strftime("%Y_%m_%d_%H_%M_%S")

source_study = optuna.load_study(
    study_name=config_ini.get("PV", "source_study"),
    storage="sqlite:///linac_study20231212.db"
)

# --- Objective function ---
study = optuna.create_study(
    # sampler=optuna.samplers.TPESampler(),
    # sampler=optuna.samplers.CmaEsSampler(source_trials=source_study.trials),
    # sampler=optuna.samplers.CmaEsSampler(),
    direction="maximize",
    # sampler=optuna.integration.BoTorchSampler(),
    sampler=Hitohudebayes.HitohudebayesSampler(),
    study_name="{}".format(filename),
    storage="sqlite:///linac_study20231214.db",
)
if config_ini.get("PV", "Initialization") == "enqueue" :
    for trial in sorted(source_study.trials, key=lambda t: t.value)[90:]:
        study.enqueue_trial(trial.params)
        print(trial.params)
elif config_ini.get("PV", "Initialization") == "param_init" :
    study.enqueue_trial(xinit)
    print(xinit)
else : print("no initialization")


peakhold = []


class StopWhenTrialKeepBeingUnupdated:  # しばらく更新しなくなったら止める
    def __init__(self, distance: int, unupdatedlength: int, threshold: float):
        self.distance = distance
        self.unupdatedlength = unupdatedlength
        self.threshold = threshold

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.Trial) -> None:
        peakhold.append(study.best_value)

        if trial.number > self.distance:
            if (
                abs(study.best_value - peakhold[trial.number - self.unupdatedlength])
                < self.threshold
            ):
                print(
                    "{0}トライアル以降の{1}トライアルで値の更新が{2}以下だったので打ち切ります".format(
                        trial.number - self.unupdatedlength,
                        self.unupdatedlength,
                        self.threshold,
                    )
                )

                # --- Best results ---
                print("- Best objective value:" + str(study.best_value))
                best_params = study.best_params
                for i, val in enumerate(best_params):
                    xpv[i].put(best_params[val])
                    print(f"- Best {i} parameter: {best_params[val]}")
                time.sleep(0.1)
                print(
                    "Re-get the best objective value to use these parameters : "
                    + str(caget(ypvname[0]))
                )
                f.close()
                study.stop()
                


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

distance = int(config_ini.get("PV", "distance"))
unupdatedlength = int(config_ini.get("PV", "unupdatedlength"))
threshold = float(config_ini.get("PV", "threshold"))

stopper = StopWhenTrialKeepBeingUnupdated(
    distance, unupdatedlength, threshold
)  # instanceを生成

X = [0 for i in range(len(xpv))]
dx = [0 for i in range(len(xpv))]
nstep = [0 for i in range(len(xpv))]
dstep = [0 for i in range(len(xpv))]
Xold = list(xinit.values())  # 初期値を代入

n_trials = int(config_ini.get("PV", "n_trials"))

now = datetime.datetime.now()
filepath = str(config_ini.get("PV", "filepath"))

os.makedirs(filepath, exist_ok=True)

filename = filepath + "log_" + now.strftime("%Y_%m_%d_%H_%M_%S") + ".csv"

f = open(filename, "a")
writer = csv.writer(f)


fig1, ax1 = plt.subplots(1, 1, figsize=[12, 8])
plt.tick_params(labelsize=30)
ax1.grid(which="major", color="gray", linestyle="-")
ax1.set_xlabel("Trial", fontsize=30)
ax1.set_ylabel("Beam charge (nC)", fontsize=30)
#ax1.set_yscale('log')

xdata,y1data,y2data = [],[],[]

for n in range(n_trials):
    trial = study.ask()

    for i in range(nxr):
        X[i] = trial.suggest_float(xpvname[i], xrange[i][0], xrange[i][1])
    Hitohudebayes.Xcopy = X

    for i in range(nxr):
        dx[i] = X[i] - Xold[i]  # i回目から(i-1)回目の差分
        nstep[i] = int(abs(dx[i]) / xstep[i]) + 1  # 差分から、何点を間に挟むか
        dstep[i] = dx[i] / nstep[i]  # 差分を間に挟む点で割った数
        Xold[i] = X[i]  # Xoldを更新

    for j in range(1, max(nstep) + 1):
        for k in range(len(xpv)):
            if j <= nstep[k]:
                caput(xpvname[k], dstep[k] * nstep[k] + Xold[k])
                print("{0}トライアルのx{1}を{2}/{3}分割目".format(n, k, j, nstep[k]))
        time.sleep(evalsleep / 1000.0)

    evaluate_value = measurement(
        X
    )  # tellの下のprint中のformat{1}に直接Test_fun(X)と書くとTest_fun(X)をもう一回計算してしまうため置いた。
    study.tell(trial, evaluate_value)

    xdata.append(n)  # trial回数
    y1data.append(evaluate_value)  # 評価値
    y2data.append(study.best_value)  # peakhold
    ax1.plot(xdata, y1data, color="tab:blue")  # 線を描写
    ax1.scatter(xdata, y1data, color="tab:blue", s=100)  # 点を描写
    ax1.plot(xdata, y2data, color="tab:orange")
    ax1.scatter(xdata, y2data, color="tab:orange", s=100)

    writer.writerow(
        [n, evaluate_value, study.best_value] + X + list(study.best_params.values())
    )

    plt.pause(0.001)  #plot.showと違い、リアルタイムで描写してくれる

    #stopper(study, trial)  # 変化幅が小さければ止める

    print(
        "{0}トライアルでの評価値は{1}、その時のパラメータは{2}。\nまた、最適値は{3}、その時のパラメータは{4}".format(
            n, evaluate_value, X, study.best_value, study.best_params
        )
    )

f.close()

# --- Best results ---
print("- Best objective value:" + str(study.best_value))
best_params = study.best_params
for i, val in enumerate(best_params):
    xpv[i].put(best_params[val])
    print(f"- Best {i} parameter: {best_params[val]}")
time.sleep(1.1)

lastresult = 0
for i in range (3):
    time.sleep(1.1)    
    lastresult += caget(ypvname[0])
    print(lastresult)

print(
    "Re-get the best objective value to use these parameters : "
    + str(lastresult/3.)
)


base_columns = ["Trial", "evaluate_value", "peakhold"]
x_columns = [f"x{i}" for i in range(nxr)]
xmax_columns = [f"x{i}max" for i in range(nxr)]

fig2, ax2 = plt.subplots(nxr, nxr, figsize=[18, 15])


data = pd.read_csv(
    filename, names=base_columns + x_columns + xmax_columns, encoding="SHIFT-JIS"
)

print(data)
projection = []

for i in range(nxr):
    for j in range(nxr):
        scatter_plot = ax2[(nxr - 1) - j, i].scatter(
            data[f"x{i}"],
            data[f"x{j}"],
            c=data["evaluate_value"],
            cmap="coolwarm",
            marker="o",
            label=f"[ {i}, {j} ] projection graph",
            #norm = LogNorm(),
        )
        
        ax2[(nxr - 1) - j, i].plot(data[f"x{i}"], data[f"x{j}"], c="grey", alpha=0.5)
        ax2[(nxr - 1) - j, i].legend()
        ax2[(nxr - 1) - j, i].grid()
        ax2[(nxr - 1) - j, i].set_xlabel("x" + str(i))
        ax2[(nxr - 1) - j, i].set_ylabel("x" + str(j))

projection.append(scatter_plot)
fig2.colorbar(projection[len(projection) - 1], ax=ax2)
plt.show()

fig1.savefig(filepath + "peakhold" + now.strftime("%Y_%m_%d_%H_%M_%S") + ".png")
fig2.savefig(filepath + "projection "+ now.strftime("%Y_%m_%d_%H_%M_%S") + ".png")

# スレッドを分けるやつ。今はいらない。
# stop_threads = False
# t_measure = threading.Thread(target = measure)
# t_measure.start()

# stop_threads = True
# t_measure.join()


# optuna.visualization.plot_param_importances(study).show()
# plot_optimization_history(study).show()
# plot_parallel_coordinate(study).show()
# plot_contour(study).show()

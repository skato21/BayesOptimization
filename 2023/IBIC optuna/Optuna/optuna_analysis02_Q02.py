import optuna
import numpy as np

from matplotlib import pyplot as plt

import logging
import sys
import configparser

#--- Load configuration ---
args = sys.argv
config_ini = configparser.ConfigParser()
config_ini.read(args[1])

color = ['black','red','green','blue','magenta']

fig = plt.figure(figsize=[9,6])
ax = fig.add_subplot(1,1,1)

for i in range(5):
    study = optuna.load_study(
        # study_name=config_ini.get('BoTorch0', 'trial{0}'.format(i)),
        # study_name=config_ini.get('TPE0', 'trial{0}'.format(i)),
        study_name=config_ini.get('CMAES0', 'trial{0}'.format(i)),
        storage="sqlite:///optuna_dispersion02_Q1.db"
    )

    x = []
    y = []
    ypre = 0
    travel = 0
    for trial in study.get_trials():
        x.append(trial.number)
        travel += abs(trial.value - ypre)
        ypre = trial.value
        y.append(travel)

    if i==0:
        ax.plot(x, y, alpha=0.5, linestyle="solid", color=color[i], label='Random')
    else:
        ax.plot(x, y, alpha=0.5, linestyle="solid", color=color[i])

for i in range(5):
    study = optuna.load_study(
        # study_name=config_ini.get('BoTorch1', 'trial{0}'.format(i)),
        # study_name=config_ini.get('TPE1', 'trial{0}'.format(i)),
        study_name=config_ini.get('CMAES1', 'trial{0}'.format(i)),
        storage="sqlite:///optuna_dispersion02_Q1.db"
    )

    x = []
    y = []
    ymax = 0
    travel = 0
    for trial in study.get_trials():
        x.append(trial.number)
        travel += abs(trial.value - ypre)
        ypre = trial.value
        y.append(travel)

    if i==0:
        ax.plot(x, y, alpha=0.5, linestyle="dashed", color=color[i], label='Previous best value')
    else:
        ax.plot(x, y, alpha=0.5, linestyle="dashed", color=color[i])

ax.set_xlabel('Trial', fontsize=30)
ax.set_ylabel('Travel distance (nC)', fontsize=30)
ax.legend(loc='upper right', frameon=False, fancybox=False, fontsize=25)
# ax.text(0, 200, "(b) TPE", fontsize=20)
# ax.text(0, 200, "(a) Bayesian optimization", fontsize=25)
ax.text(0, 200, "(b) CMA-ES", fontsize=25)
plt.ylim([0, 300])
plt.grid(which = "major" , color = "lightgray" , linestyle = "-")
plt.tick_params(labelsize=30)
plt.subplots_adjust(left=0.16, right=0.98, bottom=0.15, top=0.97)
plt.show()

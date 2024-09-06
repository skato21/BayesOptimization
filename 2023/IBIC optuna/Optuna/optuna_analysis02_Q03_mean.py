import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

import numpy as np

from matplotlib import pyplot as plt

import logging
import sys
import configparser

# --- Load configuration ---
args = sys.argv
config_ini = configparser.ConfigParser()
config_ini.read(args[1])

color = ['black', 'red', 'green', 'blue', 'magenta']

fig = plt.figure(figsize=[9,6])
ax = fig.add_subplot(1, 1, 1)
x = ['$I_{x2}$','$I_{y2}$', '$I_{x1}$', '$I_{y1}$', '$I_{x0}$', '$I_{y0}$', ]

y0 = [0] * 6
for i in range(5):
    study = optuna.load_study(
        # study_name=config_ini.get('BoTorch0', 'trial{0}'.format(i)),
        # study_name=config_ini.get('TPE0', 'trial{0}'.format(i)),
        study_name=config_ini.get('CMAES0', 'trial{0}'.format(i)),
        storage="sqlite:///optuna_dispersion02_Q1.db"
    )

    importances = optuna.importance.get_param_importances(study=study)

    for key, value in importances.items():
        index = int(key[1])
        y0[index] += value/5.

ax.scatter(x, y0, s=100, alpha=0.5, marker='o', linewidths=2,
           facecolor='None', edgecolors='black', label='Random')

y1 = [0] * 6
for i in range(5):
    study = optuna.load_study(
        # study_name=config_ini.get('BoTorch1', 'trial{0}'.format(i)),
        # study_name=config_ini.get('TPE1', 'trial{0}'.format(i)),
        study_name=config_ini.get('CMAES1', 'trial{0}'.format(i)),
        storage="sqlite:///optuna_dispersion02_Q1.db"
    )

    importances = optuna.importance.get_param_importances(study=study)

    for key, value in importances.items():
        index = int(key[1])
        y1[index] += value/5.

ax.scatter(x, y1, s=100, alpha=0.5, marker='s', linewidths=2,
           facecolor='None', edgecolors='red', label='Previous best value')

ax.set_ylabel('Importance', fontsize=30)
ax.legend(loc='upper right', frameon=False, fancybox=False, fontsize=25)
# ax.text(-0.1, 0.85, "(a) Bayesian optimization", fontsize=25)
# ax.text(-0.1, 1, "(b) TPE", fontsize=20)
ax.text(-0.1, 1, "(b) CMA-ES", fontsize=25)
plt.ylim([0, 1.30])
plt.grid(which = "major" , color = "lightgray" , linestyle = "-")
#plt.xticks(rotation=70)
plt.tick_params(labelsize=30)
plt.subplots_adjust(left=0.18, right=0.98, bottom=0.10, top=0.97)
plt.show()

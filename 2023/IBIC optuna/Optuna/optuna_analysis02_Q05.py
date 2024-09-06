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

#--- Load configuration ---
args = sys.argv
config_ini = configparser.ConfigParser()
config_ini.read(args[1])

color = ['black','red','green','blue','magenta']

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = ['PX_R0_01', 'PY_R0_01', 'PX_R0_02', 'PY_R0_02', 'PX_A4_4', 'PY_A4_4']

for i in range(5):
    study = optuna.load_study(
        # study_name=config_ini.get('BoTorch0', 'trial{0}'.format(i)),
        study_name=config_ini.get('TPE0', 'trial{0}'.format(i)),
        # study_name=config_ini.get('CMAES0', 'trial{0}'.format(i)),
        storage="sqlite:///optuna_dispersion02_Q1.db"
    )

    info = optuna.visualization._edf._get_edf_info(study)

    for study_name, y_values in info.lines:
        if i==0:
            ax.plot(info.x_values, y_values, alpha=0.5, linestyle="solid", color=color[i], label='Without enqueue')
        else:
            ax.plot(info.x_values, y_values, alpha=0.5, linestyle="solid", color=color[i])

for i in range(5):
    study = optuna.load_study(
        # study_name=config_ini.get('BoTorch1', 'trial{0}'.format(i)),
        study_name=config_ini.get('TPE1', 'trial{0}'.format(i)),
        # study_name=config_ini.get('CMAES1', 'trial{0}'.format(i)),
        storage="sqlite:///optuna_dispersion02_Q1.db"
    )

    info = optuna.visualization._edf._get_edf_info(study)

    for study_name, y_values in info.lines:
        if i==0:
            ax.plot(info.x_values, y_values, alpha=0.5, linestyle="dashed", color=color[i], label='Use enqueue')
        else:
            ax.plot(info.x_values, y_values, alpha=0.5, linestyle="dashed", color=color[i])

ax.set_xlabel('Charge (nC)', fontsize=20)
ax.set_ylabel('EDF', fontsize=20)
ax.legend(loc='upper right', frameon=False, fancybox=False, fontsize=20)
# ax.text(0.15, 1, "(a) BoTorch", fontsize=20)
ax.text(0.15, 1, "(b) TPE", fontsize=20)
# ax.text(0.15, 1, "(c) CMA-ES", fontsize=20)
plt.xlim([0, 9.9])
plt.ylim([0, 1.35])
plt.tick_params(labelsize=20)
plt.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.97)
plt.show()

# import optuna
import numpy as np
import sys
sys.path.insert(0, '/home/mitsuka/work/2023/0714/optuna')
print(sys.path)
import optuna


from matplotlib import pyplot as plt
import scienceplots

import logging
import sys
import configparser

#--- Load configuration ---
args = sys.argv
config_ini = configparser.ConfigParser()
config_ini.read(args[1])

study = optuna.load_study(
    study_name=config_ini.get('dispersion03', 'botorch1'),
    storage="sqlite:///optuna_dispersion03.db"
)

ax = optuna.visualization.matplotlib.plot_pareto_front(study)
ax.set_facecolor("white")
ax.set_xlabel('Dispersion (m)', fontsize=20, color='black')
ax.set_ylabel('Charge (nC)', fontsize=20, color='black')
ax.legend(loc='lower right', frameon=False, fancybox=False, fontsize=20)

ax.set_title("")
plt.xlim([-0.1, 5])
plt.ylim([-0.1, 10])
plt.tick_params(labelsize=20, color='black')
plt.subplots_adjust(left=0.13, right=0.98, bottom=0.15, top=0.97)
plt.tight_layout()
ax.axhline(color='black')
ax.axvline(color='black')
plt.show()

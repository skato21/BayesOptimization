import sys
sys.path.insert(0, '/home/mitsuka/work/2023/0714/optuna')
print(sys.path)
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
import plotly.express as px

#--- Load configuration ---
args = sys.argv
config_ini = configparser.ConfigParser()
config_ini.read(args[1])

for i in range(1):
    study = optuna.load_study(
        study_name=config_ini.get('BoTorch500', 'trial{0}'.format(i)),
        # study_name=config_ini.get('TPE500', 'trial{0}'.format(i)),
        storage="sqlite:///optuna_dispersion02_Q1.db"
    )

    ax = optuna.visualization.matplotlib.plot_contour(study, params=["x5","x4"], target_name='Charge (nC)')
    plt.title("", fontsize=0)
    plt.xlabel("PX_A4_4", fontsize=20)
    plt.ylabel("PY_A4_4", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.show()

    # fig = optuna.visualization.plot_contour(study, params=["x4", "x0"])
    # fig.update_layout(
    #     width=1200,
    #     height=1000,
    #     margin=dict(l=20, r=20, t=20, b=20),
    #     paper_bgcolor="White",
    #     font_color="black",
    #     font=dict(size=60)
    # )
    # fig.show()
    # df = px.data.tips()

import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

from pathlib import Path
import sys
sys.path.append('/home/mitsuka/work/2023/0714/fanova/build/lib')
from fanova import fANOVA
import fanova.visualizer

import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import numpy as np

from matplotlib import pyplot as plt

import logging
import configparser
import matplotlib

# --- Load configuration ---
args = sys.argv
config_ini = configparser.ConfigParser()
config_ini.read(args[1])

color = ['black', 'red', 'green', 'blue', 'magenta']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = ['PX_R0_01', 'PY_R0_01', 'PX_R0_02', 'PY_R0_02', 'PX_A4_4', 'PY_A4_4']
y = np.zeros((6, 6))

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=20)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=16)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=16)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)], fontsize=14)
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# for i in range(5):
for i in range(1):
    study = optuna.load_study(
        # study_name=config_ini.get('BoTorch1', 'trial{0}'.format(i)),
        # study_name=config_ini.get('TPE0', 'trial{0}'.format(i)),
        # study_name=config_ini.get('CMAES1', 'trial{0}'.format(i)),
        study_name=config_ini.get('BoTorch500', 'trial{0}'.format(i)),
        # study_name=config_ini.get('TPE500', 'trial{0}'.format(i)),
        storage="sqlite:///optuna_dispersion02_Q1.db"
    )

    importances = optuna.importance.get_param_importances(study=study)

    for key, value in importances.items():
        index = int(key[1])
        print(index, value)

    df0 = study.trials_dataframe(attrs=(['params']), multi_index=False)
    df0.to_csv("features.csv", header=False, index=False)

    df1 = study.trials_dataframe(attrs=(['value']), multi_index=False)
    df1.to_csv("responses.csv", header=False, index=False)

    # artificial dataset (here: features)
    features = np.loadtxt('features.csv', delimiter=",")
    responses = np.loadtxt('responses.csv', delimiter=",")

    # config space
    pcs = list(zip(np.min(features, axis=0), np.max(features, axis=0)))
    cs = ConfigSpace.ConfigurationSpace()
    for i in range(len(pcs)):
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "%i" % i, pcs[i][0], pcs[i][1]))

    # create an instance of fanova with trained forest and ConfigSpace
    f = fANOVA(X=features, Y=responses, config_space=cs)

    for i in range(len(pcs)):
        for j in range(len(pcs)):
            if i < j:
                best_margs = f.get_most_important_pairwise_marginals((i, j))
                for key, val in best_margs.items():
                    # y[i][j] += val/5.
                    y[i][j] += val

for i in range(1):
    study = optuna.load_study(
        # study_name=config_ini.get('BoTorch1', 'trial{0}'.format(i)),
        # study_name=config_ini.get('TPE0', 'trial{0}'.format(i)),
        # study_name=config_ini.get('CMAES1', 'trial{0}'.format(i)),
        # study_name=config_ini.get('BoTorch500', 'trial{0}'.format(i)),
        study_name=config_ini.get('TPE500', 'trial{0}'.format(i)),
        storage="sqlite:///optuna_dispersion02_Q1.db"
    )

    importances = optuna.importance.get_param_importances(study=study)

    for key, value in importances.items():
        index = int(key[1])
        print(index, value)

    df0 = study.trials_dataframe(attrs=(['params']), multi_index=False)
    df0.to_csv("features.csv", header=False, index=False)

    df1 = study.trials_dataframe(attrs=(['value']), multi_index=False)
    df1.to_csv("responses.csv", header=False, index=False)

    # artificial dataset (here: features)
    features = np.loadtxt('features.csv', delimiter=",")
    responses = np.loadtxt('responses.csv', delimiter=",")

    # config space
    pcs = list(zip(np.min(features, axis=0), np.max(features, axis=0)))
    cs = ConfigSpace.ConfigurationSpace()
    for i in range(len(pcs)):
        cs.add_hyperparameter(UniformFloatHyperparameter(
            "%i" % i, pcs[i][0], pcs[i][1]))

    # create an instance of fanova with trained forest and ConfigSpace
    f = fANOVA(X=features, Y=responses, config_space=cs)

    for i in range(len(pcs)):
        for j in range(len(pcs)):
            if i > j:
                best_margs = f.get_most_important_pairwise_marginals((i, j))
                for key, val in best_margs.items():
                    # y[i][j] += val/5.
                    y[i][j] += val

print(y)

fig, ax = plt.subplots()
im, cbar = heatmap(y, x, x, ax=ax, cmap="Blues", cbarlabel="Importance")
texts = annotate_heatmap(im, valfmt="{x:.2f}")

fig.tight_layout()
plt.show()

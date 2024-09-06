#!/usr/bin/env python
# coding: utf-8

# ## Basic Bayesian Optimization
# In this tutorial we demonstrate the use of Xopt to preform Bayesian Optimization on a
#  simple test problem.

# ## Define the test problem
# Here we define a simple optimization problem, where we attempt to minimize the sin
# function in the domian [0,2*pi]. Note that the function used to evaluate the
# objective function takes a dictionary as input and returns a dictionary as the output.

# In[ ]:






# 設定ファイルを扱うモジュールをインポート
import configparser

# ConfigParserのインスタンス（特定の機能を持った変数）を取得
config = configparser.ConfigParser()

# 用意したconfig_1.iniを読み出し
config.read("config_1.ini")

# 変数Config_1の中から、"BASE"セクションの"speed"と"weight"項目の内容を取り出し
cfg_num = config["BASE"]["number_of_parametor"]
cfg_x0min = config["BASE"]["minimum_of_x0"]
cfg_x0max = config["BASE"]["maximum_of_x0"]
cfg_x1min = config["BASE"]["minimum_of_x1"]
cfg_x1max = config["BASE"]["maximum_of_x1"]
cfg_x2min = config["BASE"]["minimum_of_x2"]
cfg_x2max = config["BASE"]["maximum_of_x2"]
cfg_x3min = config["BASE"]["minimum_of_x3"]
cfg_x3max = config["BASE"]["maximum_of_x3"]
cfg_x4min = config["BASE"]["minimum_of_x4"]
cfg_x4max = config["BASE"]["maximum_of_x4"]
cfg_x5min = config["BASE"]["minimum_of_x5"]
cfg_x5max = config["BASE"]["maximum_of_x5"]


# 変数の内容を出力
#print("cfg_read_1 =", cfg_read_1)
#print("cfg_read_2 =", cfg_read_2)

# 変数の内容を変更
#cfg_read_1 = int(cfg_read_1)

# configの各項目に上書き
#config["BASE"]["speed"] = str(cfg_read_1)

# config_1.iniファイルに上書き
#with open("config_1.ini", "w") as file:
#    config.write(file)






from xopt.vocs import VOCS
import numpy as np
import math
from matplotlib.colors import LogNorm 

# difine dimension of function
N = int(cfg_num)

# define variables and function objectives
vocs = VOCS(
    variables={ "x0": [float(cfg_x0min),float(cfg_x0max)] ,"x1": [float(cfg_x1min),float(cfg_x1max)] , "x2": [float(cfg_x2min),float(cfg_x2max)] , "x3": [float(cfg_x3min),float(cfg_x3max)] , "x4": [float(cfg_x4min),float(cfg_x4max)] , "x5": [float(cfg_x5min),float(cfg_x5max)]},
    objectives={"f": "MAXIMIZE"},
)

# In[ ]:

from epics import caget , caput
import time


def Colville_EPICS_fun(inputs):
    caput ('LIiMG:PX_R0_61:IWRITE:KBP', inputs["x0"])
    caput ('LIiMG:PY_R0_61:IWRITE:KBP', inputs["x1"])
    caput ('LIiMG:PX_R0_63:IWRITE:KBP', inputs["x2"])
    caput ('LIiMG:PY_R0_63:IWRITE:KBP', inputs["x3"])
    caput ('LIiMG:PX_13_5:IWRITE:KBP', inputs["x4"])
    caput ('LIiMG:PY_13_5:IWRITE:KBP', inputs["x5"])
    time.sleep(2.0)
    return{"f": caget('LIiBM:SP_16_5_1:ISNGL:KBP')}


# ## Create Xopt objects
# Create the evaluator to evaluate our test function and create a generator that uses
# the Upper Confidence Bound acqusition function to perform Bayesian Optimization.

# In[ ]:


from xopt.evaluator import Evaluator
from xopt.generators.bayesian import UpperConfidenceBoundGenerator, ExpectedImprovementGenerator
from xopt import Xopt
from xopt.generators import get_generator_and_defaults
ucb_gen, ucb_options = get_generator_and_defaults("upper_confidence_bound")

evaluator = Evaluator(function=Colville_EPICS_fun)
#generator = UpperConfidenceBoundGenerator(vocs)
#generator_options = ucb_gen.default_options()
#generator_options.acq.beta = 2.0
generator = ExpectedImprovementGenerator(vocs)
#generator = BayesianExplorationGenerator(vocs)
X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)


# ## Generate and evaluate initial points
# To begin optimization, we must generate some random initial data points. The first call
# to `X.step()` will generate and evaluate a number of randomly points specified by the
#  generator. Note that if we add data to xopt before calling `X.step()` by assigning
#  the data to `X.data`, calls to `X.step()` will ignore the random generation and
#  proceed to generating points via Bayesian optimization.

# In[ ]:


# print initial number of points to be generated
print(X.generator.options.n_initial)

# call X.step() to generate + evaluate initial points
X.step()

# inspect the gathered data
X.data


# ## Do bayesian optimization steps
# To perform optimization we simply call `X.step()` in a loop. This allows us to do
# intermediate tasks in between optimization steps, such as examining the model and
# acquisition function at each step (as we demonstrate here).

# In[ ]:

import csv
import datetime
import os
import csv

now = datetime.datetime.now()
filepath = '/nfs/sadstorage-users/mitsuka/work/2023/0614/positron_study_log/'


os.makedirs(filepath, exist_ok=True)

filename =  filepath + 'log_' + now.strftime('%Y_%m_%d_%H_%M_%S') + '.csv'

f = open(filename, 'a')
writer = csv.writer(f)



import torch
import matplotlib.pyplot as plt

fig1,ax1 = plt.subplots(1,1,figsize=[15,10],facecolor = 'lightgray')
ax1.grid(which = "major" , color = "black" , linestyle = "-")
ax1.set_xlabel("loop count")
ax1.set_ylabel("Optimization value")


x = {}
y = {}
ymax = 0

for i in range(100):
    # get the Gaussian process model from the generator
    model = X.generator.train_model()

    # get acquisition function from generator
    acq = X.generator.get_acquisition(model)

    # calculate model posterior and acquisition function at each test point
    # NOTE: need to add a dimension to the input tensor for evaluating the
    # posterior and another for the acquisition function, see
    # https://botorch.org/docs/batching for details
    # NOTE: we use the `torch.no_grad()` environment to speed up computation by
    # skipping calculations for backpropagation

    X.step()

    x[i] = i+1
    y[i] = X.data["f"][i+4]

    if  y[i] > ymax:
        ymax =  y[i]
        x0max = X.data["x0"][i+4]
        x1max = X.data["x1"][i+4]
        x2max = X.data["x2"][i+4]
        x3max = X.data["x3"][i+4]
        x4max = X.data["x4"][i+4]
        x5max = X.data["x5"][i+4]

    
    writer.writerow([i,y[i],X.data["x0"][i+4],X.data["x1"][i+4],X.data["x2"][i+4],X.data["x3"][i+4],X.data["x4"][i+4],X.data["x3"][i+4],ymax,x0max,x1max,x2max,x3max,x4max,x5max])

    #print('Result',i, y[i], X.data["x0"][i+4], X.data["x1"][i+4], X.data["x2"][i+4], X.data["x3"][i+4], ymin, x0min, x1min, x2min, x3min)

    ax1.plot(x[i] ,y[i], "C0o" , markersize = 10)
    ax1.plot(x[i] ,ymax, "C1o" , markersize = 10)

    plt.pause(1)


f.close()



# In[ ]:


fig2,ax2 = plt.subplots(N,N,figsize=[18,15],facecolor = 'lightgray')

sc = list()

for i in range (N):
    for j in range (N-i):



        sc.append(ax2[j,N-1-i].scatter(X.data["x" + str(j)], X.data["x" + str(N-1-i)] , c = X.data["f"] ,cmap = "coolwarm",   marker='o' , label = "[ " + str(j) + ", " + str(N-1-i) + " ] projection graph" ))

        #norm = LogNorm(),
        ax2[j,N-1-i].legend()
        ax2[j,N-1-i].grid()
        ax2[j,N-1-i].set_xlabel("x" + str(N-1-i))
        ax2[j,N-1-i].set_ylabel("x" + str(j))
        
fig2.colorbar(sc[len(sc)-1],ax=ax2)
plt.show()

#fig1.savefig("ax1 " + now.strftime('%Y_%m_%d_%H_%M_%S'))
#fig2.savefig("fig2 " + now.strftime('%Y_%m_%d_%H_%M_%S'))
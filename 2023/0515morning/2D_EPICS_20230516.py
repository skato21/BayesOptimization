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

from xopt.vocs import VOCS
import numpy as np
import math

# difine dimension of function
N = 2 

# define variables and function objectives
vocs = VOCS(
    variables={ "x0" : [-5.5,-1.5], "x1" : [-1.5,3.5]},
    objectives={"f": "MAXIMIZE"},
)


# In[ ]:

from epics import caget , caput
import time

def Colville_EPICS_fun(inputs):
    caput ('LIiMG:PX_R0_61:IWRITE:KBP', inputs["x0"])
    caput ('LIiMG:PY_R0_61:IWRITE:KBP', inputs["x1"])
    #caput ('TEST:X2', inputs["x2"])
    #caput ('TEST:X3', inputs["x3"])
    time.sleep(2.0)
    return{"f": caget('LIiBM:SP_16_5_1:ISNGL:KBP')}


# ## Create Xopt objects
# Create the evaluator to evaluate our test function and create a generator that uses
# the Upper Confidence Bound acqusition function to perform Bayesian Optimization.

# In[ ]:


from xopt.evaluator import Evaluator
from xopt.generators.bayesian import UpperConfidenceBoundGenerator, ExpectedImprovementGenerator, BayesianExplorationGenerator
from xopt import Xopt

evaluator = Evaluator(function=Colville_EPICS_fun)
#generator = UpperConfidenceBoundGenerator(vocs)
#generator = ExpectedImprovementGenerator(vocs)
generator = BayesianExplorationGenerator(vocs)
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


import torch
import matplotlib.pyplot as plt

fig1,ax1 = plt.subplots(1,1,figsize=[15,10],facecolor = 'lightgray')

x = {}
y = {}
ymax = 0
x0max = 0
x1max = 0

for i in range(50):
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

    print('Result',i, y[i], ymax, x0max, x1max)

    ax1.plot(x[i] ,y[i], "C0o" , markersize = 10)
    ax1.plot(x[i] ,ymax, "C1o" , markersize = 10)

    plt.pause(1)

ax1.grid(which = "major" , color = "black" , linestyle = "-")
ax1.set_xlabel("loop count")
ax1.set_ylabel("Optimization value")


# In[ ]:


fig2,ax2 = plt.subplots(N,N,figsize=[18,15],facecolor = 'lightgray')

for i in range (N):
    for j in range (N-i):

        ax2[j,N-1-i].plot(X.data["x" + str(j)], X.data["x" + str(N-1-i)] , marker='o' , label = "[ " + str(j) + ", " + str(N-1-i) + " ] projection graph" )
        ax2[j,N-1-i].legend()
        ax2[j,N-1-i].grid()
        ax2[j,N-1-i].set_xlabel("x" + str(N-1-i))
        ax2[j,N-1-i].set_ylabel("x" + str(j))

plt.show()

# # do the optimization step
# X.step()


# ## Getting the optimization result
# To get the ideal point (without evaluating the point) we ask the generator to
# generate a new point.

# In[ ]:


X.generator.get_optimum()


# ## Customizing optimization
# Each generator has a set of options that can be modified to effect optimization behavior

# In[ ]:


X.generator.options.dict()


# In[ ]:


# example: add a Gamma(1.0,10.0) prior to the noise hyperparameter to reduce model noise
# (good for optimizing noise-free simulations)
X.generator.options.model.use_low_noise_prior = True


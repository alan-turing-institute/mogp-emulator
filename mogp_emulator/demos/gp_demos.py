import mogp_emulator
import numpy as np

# simple GP examples

# simulator function -- needs to take a single input and output a single number

def f(x):
    return np.exp(-np.sum((x-2.)**2, axis = -1)/2.)

# Experimental design -- requires a list of parameter bounds if you would like to use
# uniform distributions. If you want to use different distributions, you
# can use any of the standard distributions available in scipy to create
# the appropriate ppf function (the inverse of the cumulative distribution).
# Internally, the code creates the design on the unit hypercube and then uses
# the distribution to map from [0,1] to the real parameter space.

ed = mogp_emulator.LatinHypercubeDesign([(0., 5.), (0., 5.)])

# sample space

inputs = ed.sample(20)

# run simulation

targets = np.array([f(p) for p in inputs])

###################################################################################

# First example -- fit GP using MLE and Squared Exponential Kernel and predict

print("Example 1: Basic GP")

# create GP and fit using MLE

gp = mogp_emulator.GaussianProcess(inputs, targets)

gp.learn_hyperparameters()

# create 20 target points to predict

predict_points = ed.sample(10)

means, variances, derivs = gp.predict(predict_points)

for pp, m in zip(predict_points, means):
    print("Target point: {}      Predicted mean: {}".format(pp, m))

###################################################################################

# Second Example: How to change the kernel

print("Example 2: Matern Kernel")

# create GP as before

gp_matern = mogp_emulator.GaussianProcess(inputs, targets)

# manually change to Matern 5/2 kernel

from mogp_emulator.Kernel import Matern52

gp_matern.kernel = Matern52()

# continue as before

gp_matern.learn_hyperparameters()

# create 20 target points to predict

predict_points = ed.sample(10)

means, variances, derivs = gp_matern.predict(predict_points)

for pp, m in zip(predict_points, means):
    print("Target point: {}      Predicted mean: {}".format(pp, m))

###################################################################################

# Third Example: Fit Hyperparameters via MCMC sampling

print("Example 3: MCMC fitting")

# create GP as before

gp_mcmc = mogp_emulator.GaussianProcess(inputs, targets)

# fit hyperparameters with MCMC

gp_mcmc.learn_hyperparameters_MCMC(n_samples = 10000)

# create 20 target points to predict

predict_points = ed.sample(10)

# predict using MCMC samples

means, variances, derivs = gp_mcmc.predict(predict_points, predict_from_samples = True)

for pp, m in zip(predict_points, means):
    print("Target point: {}      Predicted mean: {}".format(pp, m))
import numpy as np
import mogp_emulator
from projectile import simulator, print_results

# GP example using the projectile demo on a GPU

# define some common variables

n_samples = 20
n_preds = 10

# Experimental design -- requires a list of parameter bounds if you would like to use
# uniform distributions. If you want to use different distributions, you
# can use any of the standard distributions available in scipy to create
# the appropriate ppf function (the inverse of the cumulative distribution).
# Internally, the code creates the design on the unit hypercube and then uses
# the distribution to map from [0,1] to the real parameter space.

ed = mogp_emulator.LatinHypercubeDesign([(-5., 1.), (0., 1000.)])

# sample space

inputs = ed.sample(n_samples)

# run simulation

targets = np.array([simulator(p) for p in inputs])

###################################################################################

# Basic example -- fit GP using MLE and Squared Exponential Kernel and predict

print("Example: Basic GP")

# create GP and then fit using MLE
# the only difference between this and the standard CPU implementation
# is to use the GaussianProcessGPU class rather than GaussianProcess.

gp = mogp_emulator.GaussianProcessGPU(inputs, targets)

gp = mogp_emulator.fit_GP_MAP(gp)

# create 20 target points to predict

predict_points = ed.sample(n_preds)

means, variances, derivs = gp.predict(predict_points)

print_results(predict_points, means)

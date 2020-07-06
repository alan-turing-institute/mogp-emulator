import numpy as np
from projectile import simulator, print_results
import mogp_emulator

try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

# An end-to-end tutorial illustrating model calibration using mogp_emulator

# First, we need to set up our experimental design. We would like our drag coefficient to be
# on a logarithmic scale and initial velocity to be on a linear scale. However, our simulator
# does the drag coefficient transformation for us, so we simply can specity the exponent on
# a linear scale.

# We will use a Latin Hypercube Design. To specify, we give the distribution that we would like
# the parameter to take. By default, we assume a uniform distribution between two endpoints, which
# we will use for this simulation.

# Once we construct the design, can draw a specified number of samples as shown.

lhd = mogp_emulator.LatinHypercubeDesign([(-5., 1.), (0., 1000.)])

n_simulations = 50
simulation_points = lhd.sample(n_simulations)
simulation_output = np.array([simulator(p) for p in simulation_points])

# Next, fit the surrogate GP model using MLE (MAP with uniform priors)
# Print out hyperparameter values as correlation lengths and sigma

gp = mogp_emulator.GaussianProcess(simulation_points, simulation_output)
gp = mogp_emulator.fit_GP_MAP(gp)

print("Correlation lengths = {}".format(np.sqrt(np.exp(-gp.theta[:2]))))
print("Sigma = {}".format(np.sqrt(np.exp(gp.theta[2]))))

# Validate emulator by comparing to true simulated value
# To compare with the emulator, use the predict method to get mean and variance
# values for the emulator predictions and see how many are within 2 standard
# deviations

n_valid = 10
validation_points = lhd.sample(n_valid)
validation_output = np.array([simulator(p) for p in validation_points])

predictions = gp.predict(validation_points)

print_results(validation_points, predictions.mean)

# Finally, perform history matching. Sample densely from the experimental design and
# determine which points are consistent with the data using the GP predictions
# We compute which points are "Not Ruled Out Yet" (NROY)

n_predict = 10000
prediction_points = lhd.sample(n_predict)

hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[2000., 400.])

nroy_points = hm.get_NROY()

print("Ruled out {} of {} points".format(n_predict - len(nroy_points), n_predict))

# If plotting enabled, visualize results

if makeplots:
    plt.plot(prediction_points[nroy_points,0], prediction_points[nroy_points,1], "o", label="NROY points")
    plt.plot(simulation_points[:,0], simulation_points[:,1],"o", label="Simulation Points")
    plt.xlabel("log Drag Coefficient")
    plt.ylabel("Launch velocity (m/s)")
    plt.legend()
    plt.show()
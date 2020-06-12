"""
Benchmark illustrating using MCMC fitting to estimate hyperparameters. From a set of input
data with two inputs, posterior samples are drawn for the hyperparameters. The benchmark
produces output comparing the posterior samples to the MLE solution found by minimizing
the negative log-likelihood, and shows the predictions using MLE and MCMC estiation on a set
of test points. Plots showing a histogram of the posterior samples are made if ``matplotlib``
is installed.
"""

import numpy as np
from mogp_emulator import GaussianProcess
try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

def f(x, y):
    a = 5.
    b = 0.03
    c = 4.
    d = 0.03
    e = 1.
    f = 50.
    g = 10.
    return g*np.exp(-(x-a)**2/2./c-(y-b)**2/2./d)*np.sin(e*x)**2*np.sin(f*y)**2

def get_data(n_inputs, n_predict):
    "generate data for GP fitting"
    
    assert n_inputs > 0
    assert n_predict > 0
    
    x = np.random.uniform(0., 10., size = (n_inputs,))
    y = np.random.uniform(0., 0.1, size = (n_inputs,))
    targets = f(x, y)

    inputs = np.empty((n_inputs,2))
    inputs[:,0] = x
    inputs[:,1] = y
    
    x_predict = np.random.uniform(0., 10., size = (n_predict,))
    y_predict = np.random.uniform(0., 0.1, size = (n_predict,))
    predict_targets = f(x_predict, y_predict)
    
    predict = np.empty((n_predict, 2))
    predict[:,0] = x_predict
    predict[:,1] = y_predict
    
    return inputs, targets, predict, predict_targets

def print_GP_results(testing, testing_target, gp_mean, gp_unc, mcmc_mean, mcmc_unc):
    
    print("Test point          Target     MLE Mean   MLE Var.    MCMC Mean  MCMC Variance")
    for t, tt, m, u, mm, mu in zip(testing, testing_target, gp_mean, gp_unc, mcmc_mean, mcmc_unc):
        print('{:8.6f} {:9.6f} {:9.6f} {:11.7f} {:10.7f} {:11.7f} {:10.7f}'.format(t[0], t[1], tt, m, u, mm, mu))


def fit_MCMC(n_inputs, n_predict, n_samples = 1000, thin = 0):
    "fit a 1D GP using MCMC"
        
    inputs, targets, predict, predict_targets = get_data(n_inputs, n_predict)
    
    gp = GaussianProcess(inputs, targets)
    gp.learn_hyperparameters_MCMC(n_samples, thin)
    
    gp._set_params(gp.mle_theta)
    mean, variance, _ = gp.predict(predict)
    mean_mcmc, var_mcmc, _ = gp.predict(predict, predict_from_samples = True)
    
    print("MLE Parameters:", gp.mle_theta)
    print("MCMC Means:", np.mean(gp.samples, axis = 0))
    print("MCMC Variances:", np.var(gp.samples, axis = 0))
    print()
    print("Predictions:")
    print_GP_results(predict, predict_targets, mean, variance, mean_mcmc, var_mcmc)
    
    if makeplots:
        
        plt.figure(figsize=(6,3))
    
        plt.subplot(131)
        n, bins, _ = plt.hist(gp.samples[:,0], bins=25)
        plt.plot([gp.mle_theta[0], gp.mle_theta[0]], [0, np.max(n)], label = "MLE")
        plt.xlabel('Length scale 1')
        plt.legend()
    
        plt.subplot(132)
        n, bins, _ = plt.hist(gp.samples[:,1], bins=25)
        plt.plot([gp.mle_theta[1], gp.mle_theta[1]], [0, np.max(n)])
        plt.xlabel('Length scale 2')
        plt.title("Posterior MCMC Hyperparameter Samples and MLE values")

        plt.subplot(133)
        n, bins, _ = plt.hist(gp.samples[:,2], bins=25)
        plt.plot([gp.mle_theta[2], gp.mle_theta[2]], [0, np.max(n)])
        plt.xlabel('Covariance')
    
        plt.savefig("MCMC_histogram.png", bbox_inches = "tight")


if __name__ == '__main__':
    fit_MCMC(100, 10, 10000, 0)

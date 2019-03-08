import numpy as np
from mogp_emulator import MultiOutputGP
from mogp_emulator.utils import lhd
from scipy.stats import uniform
import matplotlib.pyplot as plt

def branin_2d(x1, x2, params):
    "2D Branin function, see https://www.sfu.ca/~ssurjano/branin.html for more information"
    a, b, c, r, s, t = params
    return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1. - t)*np.cos(x1) + s
    
def generate_emulator_params(n_emulators):
    "Generate random parameters for use with 2D Branin function, ensuring outputs are similarly valued"
    
    n_emulators = int(n_emulators)
    assert n_emulators > 0
    
    x1 = np.linspace(-5., 10.)
    x2 = np.linspace(0., 15.)
    x1m, x2m = np.meshgrid(x1, x2, indexing='ij')
    
    params = np.zeros((n_emulators, 6))
    
    i = 0
    
    while i < n_emulators:
        
        a = np.random.normal(loc = 1., scale = 0.1)
        b = np.random.normal(loc = 5.1/4./np.pi**2, scale = 0.1)
        c = np.random.normal(loc = 5./np.pi, scale = 0.1)
        r = np.random.normal(loc = 6., scale = 1.)
        s = np.random.normal(loc = 10., scale = 2.)
        t = np.random.normal(loc = 1./8./np.pi, scale = 0.01)
        
        branin_vals = branin_2d(x1m, x2m, (a, b, c, r, s, t))
        
        if np.all(branin_vals >= 0.) and np.all(branin_vals <= 350.):
            params[i] = np.array([a, b, c, r, s, t])
            i += 1
    
    return params
    
def generate_input_data(n_simulations, method = "random"):
    "Generate random points x1 and x2 for evaluating the multivalued 2D Branin function"
    
    n_simulations = int(n_simulations)
    assert(n_simulations > 0)
    assert method == "random" or method == "lhd"
    
    if method == "random":
        inputs = np.zeros((n_simulations, 2))
        inputs[:,0] = np.random.uniform(low = -5., high = 10., size = n_simulations)
        inputs[:,1] = np.random.uniform(low = 0., high = 15., size = n_simulations)
    elif method == "lhd":
        inputs = lhd([uniform(loc = -5., scale = 15.), uniform(loc = 0., scale = 15.)], n_simulations)
    return inputs
    
def generate_target_data(inputs, emulator_params):
    "Generate target data for multivalued emulator benchmark"
    
    emulator_params = np.array(emulator_params)
    assert emulator_params.shape[1] == 6
    n_emulators = emulator_params.shape[0]
    
    inputs = np.array(inputs)
    assert len(inputs.shape) == 2
    assert inputs.shape[1] == 2
    n_simulations = inputs.shape[0]
    
    targets = np.zeros((n_emulators, n_simulations))
    
    for i in range(n_emulators):
        targets[i] = branin_2d(inputs[:,0], inputs[:,1], emulator_params[i])
        
    return targets

def generate_training_data(n_emulators, n_simulations):
    "Generate n_simulations input data and evaluate using n_emulators different parameter values"
    
    emulator_params = generate_emulator_params(n_emulators)
    inputs = generate_input_data(n_simulations, method = "lhd")
    targets = generate_target_data(inputs, emulator_params)
    
    return inputs, targets, emulator_params
    
def generate_test_data(n_testing, emulator_params):
    "Generate n_testing points for testing the accuracy of an emulator"
    
    testing = generate_input_data(n_testing, method = "random")
    test_targets = generate_target_data(testing, emulator_params)
    
    return testing, test_targets
    
def run_model(n_emulators, n_simulations, n_testing, processes = None):
    "Generate training data, fit emulators, and test model accuracy on random points, returning RMSE"
    
    inputs, targets, emulator_params = generate_training_data(n_emulators, n_simulations)
    
    gp = MultiOutputGP(inputs, targets)
    gp.learn_hyperparameters(processes = processes)
    
    norm_const = np.mean(targets)
    
    testing, test_targets = generate_test_data(n_testing, emulator_params)
    
    test_vals, unc, deriv = gp.predict(testing, do_deriv = False, do_unc = True, processes = processes)
    
    return (np.sqrt(np.sum((test_vals - test_targets)**2)/float(n_emulators)/float(n_testing))/norm_const,
            np.sqrt(np.sum(unc**2)/float(n_emulators)/float(n_testing))/norm_const**2)
    
def plot_model_errors(n_emulators, n_testing, simulation_list, process_list = [None], n_iter = 10):
    "Makes plot showing accuracy of emulator as a function of n_simulations"
    
    n_simtrials = len(simulation_list)
    
    errors = np.zeros((n_simtrials, n_iter))
    uncs = np.zeros((n_simtrials, n_iter))
    
    for processes in process_list:
        for iteration in range(n_iter):
            for sim_index in range(n_simtrials):
                errors[sim_index, iteration], uncs[sim_index, iteration] = run_model(n_emulators, simulation_list[sim_index], n_testing, processes)
    
    error = np.mean(errors, axis = -1)
    unc = np.mean(uncs, axis = -1)
    
    print("\n")
    print("Convergence test results:")
    print("Num. simulations   Average prediction RMSE")
    for sim, err in zip(simulation_list, error):
        print('{:19}{}'.format(str(sim), str(err)))
        
    print("\n")
    print("Num. simulations   Average prediction variance")
    for sim, un in zip(simulation_list, unc):
        print('{:19}{}'.format(str(sim), str(un)))
    
    plt.figure(figsize=(4,3))
    plt.semilogy(simulation_list, error,'-o')
    plt.xlabel('Number of simulations')
    plt.ylabel('Average prediction RMSE')
    plt.title('Error for '+str(n_testing)+' predictions\nusing '+str(n_emulators)+' 2D Branin functions')
    plt.savefig('branin_2d_error.png',bbox_inches='tight')
    
    plt.figure(figsize=(4,3))
    plt.semilogy(simulation_list, unc,'-o')
    plt.xlabel('Number of simulations')
    plt.ylabel('Average prediction variance')
    plt.title('Uncertainty for '+str(n_testing)+' predictions\nusing '+str(n_emulators)+' 2D Branin functions')
    plt.savefig('branin_2d_unc.png',bbox_inches='tight')
    
if __name__ == '__main__':
    plot_model_errors(8, 100, [int(x) for x in np.linspace(10., 30., 11)], process_list = [4], n_iter = 10)
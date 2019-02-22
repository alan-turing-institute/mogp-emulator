import numpy as np
from mogp_emulator import MultiOutputGP
import h5py
from time import time
import matplotlib.pyplot as plt

def load_tsunami_data(filename):
    "loads tsunami data from .mat file"
    
    f = h5py.File(filename)
    
    tc = np.squeeze(np.array(f['tc']))
    xc = np.squeeze(np.array(f['xc']))
    yc = np.squeeze(np.array(f['yc']))
    
    return tc, xc, yc
    
def create_GP_tsunami_data(tc, xc, yc, n_emulators):
    "transforms arrays from tsunami data .mat file into format for multi-output emulator"
    
    # get inputs (same for all emulators) from first output
    
    first_datapoint = (xc == xc[0])
    inputs = np.transpose(tc[:,first_datapoint])
    
    assert(inputs.shape == (210,14))
    
    n_emulators = int(n_emulators)
    assert(n_emulators > 0)
    assert(n_emulators <= 894)
    
    # function to correct for bad data in input data
    
    def n_to_index(n):
        if n == 0:
            return 0
        elif n <= 3:
            return n + 1
        else:
            return n + 4
    
    targets = np.zeros((n_emulators, 210))
    start = 100
    
    for i in range(start, start + n_emulators):
        flag = (xc == xc[n_to_index(i)])
        targets[i - start] = yc[flag]
    
    return inputs, targets
 
def fit_emulators(filename, n_emulators, processes = None):
    "load data and fit emulators for tsunami data, returning the time required to fit the emulators"
     
    tc, xc, yc = load_tsunami_data(filename)
    inputs, targets = create_GP_tsunami_data(tc, xc, yc, n_emulators)
    
    gp = MultiOutputGP(inputs, targets)
    
    start_time = time()
    gp.learn_hyperparameters(processes = processes)
    finish_time = time()
    
    return finish_time - start_time

def make_scaling_plot(n_emulators_list, process_list):
    "create scaling plot showing time for fitting tsunami emulators"
    
    execution_times = []

    for n_emulators in n_emulators_list:
        emulator_execution_times = []
        for processes in process_list:
            emulator_execution_times.append(fit_emulators('gpgpgputestcase.mat', n_emulators, processes = processes))
        execution_times.append(np.array(emulator_execution_times))
    
    print("\n")
    print("Num. Emulators    Num. Processors    Execution Time (s)   Execution Time per Emulator (s)")
    for n_emulators, exec_times in zip(n_emulators_list, execution_times):
        for process, exec_time in zip(process_list, exec_times):
            print("{:18}{:19}{:21}{}".format(str(n_emulators), str(process), str(exec_time), str(exec_time/float(n_emulators))))
    
    fig = plt.figure(figsize=(4,3))
    for n_emulators, exec_times in zip(n_emulators_list, execution_times):
        plt.plot(process_list, exec_times/float(n_emulators), '-o', label = str(n_emulators)+" emulators")
    plt.legend()
    plt.xlabel('Number of processes')
    plt.ylabel('Execution time per emulator (s)')
    plt.title("Benchmark on tsunami simulation data")
    plt.savefig('tsunami_scaling.png', bbox_inches='tight')

if __name__ == "__main__":
    n_emulators_list = [8, 16, 32, 64]
    process_list = [1, 2, 4, 8]
    make_scaling_plot(n_emulators_list, process_list)

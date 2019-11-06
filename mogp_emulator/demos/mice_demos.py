import mogp_emulator
import numpy as np

# simple MICE examples with 2 inputs

# simulator function -- needs to take a single input and output a single number

def f(x): 
    return np.exp(-np.sum((x-2.)**2, axis = -1)/2.)
    
# Base design -- requires a list of parameter bounds if you would like to use
# uniform distributions. If you want to use different distributions, you
# can use any of the standard distributions available in scipy to create
# the appropriate ppf function (the inverse of the cumulative distribution).
# Internally, the code creates the design on the unit hypercube and then uses
# the distribution to map from [0,1] to the real parameter space.

ed = mogp_emulator.LatinHypercubeDesign([(0., 5.), (0., 5.)])

###################################################################################

# first example -- run entire design internally within the MICE class.

# first argument is base design (required), second is simulator function (optional,
# but required if you want the code to run the simualtions internally)

# Other optional arguments include: 
# n_samples (number of sequential design steps, optional, default is not specified
# meaning that you will specify when running the sequential design)
# n_init (size of initial design, default 10)
# n_cand (number of candidate points, default is 50)
# nugget (nugget parameter for design GP, default is to set adaptively)
# nugget_s (nugget parameter for candidate GP, default is 1.) 

md = mogp_emulator.MICEDesign(ed, f, n_samples = 20, n_init = 5, n_cand = 100)

md.run_sequential_design()

# get design and outputs

inputs = md.get_inputs()
targets = md.get_targets()

print("Example 1:")
print("Design inputs:\n", inputs)
print("Design targets:\n", targets)
print()

###################################################################################

# second example: run design manually

md2 = mogp_emulator.MICEDesign(ed, n_init = 5, n_cand = 100)

init_design = md2.generate_initial_design()

print("Example 2:")
print("Initial design:\n", init_design)

# run initial points manually

init_targets = np.array([f(s) for s in init_design])

# set initial targets

md2.set_initial_targets(init_targets)

# run 20 sequential design steps

for d in range(20):
    next_point = md2.get_next_point()
    next_target = f(next_point)
    md2.set_next_target(next_target)
    
# look at design and outputs

inputs = md2.get_inputs()
targets = md2.get_targets()

print("Final inputs:\n", inputs)
print("Final targets:\n", targets)

gp = mogp_emulator.GaussianProcess(inputs, targets)
gp.learn_hyperparameters()

testing = ed.sample(200000)

mean, unc, _ = gp.predict(testing)

print(mean)
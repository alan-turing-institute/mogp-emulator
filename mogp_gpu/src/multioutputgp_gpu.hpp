#ifndef MULTIOUTPUTGP_GPU_HPP
#define MULTIOUTPUTGP_GPU_HPP

/*
This file contains the C++ implementation of the Multi-Output Gaussian Process,
which is essentially a container of DenseGP_GPU instances, each of which is a Gaussian Process.

The key methods of the MultiOutputGP_GPU class are:
 
  predict_batch:  make a prediction on a set of testing points
  predict_variance_batch: make a prediction on a set of testing points, including variance
  predict_deriv: get the derivative of prediction on a set of testing points.
These in turn use CUDA kernels defined in the file cov_gpu.cu
*/

#include <iostream>

#include <algorithm>
#include <string>
#include <sstream>
#include <assert.h>
#include <stdexcept>

#include "util.hpp"
#include "kernel.hpp"
#include "meanfunc.hpp"
#include "densegp_gpu.hpp"

//////////////////////////////////////////////////////////////////////////////////////
////////////////////////// The MOGP class ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

// By convention, members ending "_d" are allocated on the device
class MultiOutputGP_GPU {

    // inputs
    mat inputs;

    // targets
    std::vector<vec> targets;

    // testing size
    unsigned int testing_size;

    // what Kernel are we using?
    kernel_type kern_type;

    // is the nugget adapted, fixed, or fitted?
    nugget_type nug_type;

    // how big is the nugget?
    double nug_size;

    // pointer to the Kernel
//    BaseKernel* kernel;

    // pointer to mean function
    BaseMeanFunc* meanfunc;

    std::vector<DenseGP_GPU*> emulators;

public:

    mat get_inputs(void) const
    {
        return inputs;
    }

    vec get_targets_at_index(int index) const
    {
        return targets[index];
    }

    std::vector<vec> get_targets(void) const
    {
        return targets;
    }

    int get_D(void) const
    {
        return inputs.cols();
    }

    int n_emulators(void) const
    {
        return emulators.size();
    }

    // make a single prediction per emulator(mainly for testing - most use-cases will use predict_batch or predict_deriv_batch)
    vec predict(mat_ref testing)
    {
        vec results(emulators.size());
        #pragma omp parallel for
        for (unsigned int i=0; i< emulators.size(); ++i) {
            results[i] = emulators[i]->predict(testing);
        }
        return results;
    }

    // variance of a single prediction (mainly for testing - most use-cases will use predict_variance_batch)
    double predict_variance(mat_ref testing, vec_ref var)
    {
        
        return 0.;
    }

    // Use the GP emulators to calculate a prediction on testing points, without calculating variance or derivative
    void predict_batch(mat_ref testing, mat_ref results)
    {

        #pragma omp parallel for
        for (unsigned int i=0; i< emulators.size(); ++i) {
            
            emulators[i]->predict_batch(testing, results.row(i));
        }
        
    }

    // Use the GP emulators to calculate a prediction on testing points, also calculating variance
    void predict_variance_batch(mat_ref testing, mat_ref means, mat_ref vars)
    {
        #pragma omp parallel for
        for (unsigned int i=0; i< emulators.size(); ++i) {
            
            emulators[i]->predict_variance_batch(testing, means.row(i), vars.row(i));
        }
     
    }

    // Use the GP emulator to calculate derivative of  prediction on testing points
    void predict_deriv(mat_ref testing, std::vector<mat_ref> results)
    {
       #pragma omp parallel for
        for (unsigned int i=0; i< emulators.size(); ++i) {
            emulators[i]->predict_deriv(testing, results[i]);
        }
    }

    void fit(mat_ref thetas) {
        #pragma omp parallel for
        for (unsigned int i=0; i< emulators.size(); ++i) {      
            emulators[i]->fit(thetas.row(i));
        }   
    }


    void fit_emulator(unsigned int index, vec_ref theta) {
        emulators.at(index)->fit(theta);
    }

    void create_emulators() {
        unsigned int testing_size_per_emulator = testing_size / targets.size();
        for (auto targ : targets) {
            // emulators will all have same starting parameters apart from targets
            emulators.push_back(new DenseGP_GPU(
                inputs, 
                targ, 
                testing_size_per_emulator, 
                meanfunc, 
                kern_type, 
                nug_type, 
                nug_size)
            );
           // emulators.push_back(new DummyThing());
        }
    }

    DenseGP_GPU* get_emulator(unsigned int index) {
   // DummyThing* get_emulator(unsigned int index) {
        return emulators.at(index);
    }

    // constructor
    MultiOutputGP_GPU(mat_ref inputs_,
	      std::vector<vec>& targets_,
	      unsigned int testing_size_,
          BaseMeanFunc* mean_,
          kernel_type kern_,
          nugget_type nugtype_,
          double nugsize_)
        : inputs(inputs_)
        , targets(targets_)
        , testing_size(testing_size_)
        , meanfunc(mean_)
        , kern_type(kern_)
        , nug_type(nugtype_)
        , nug_size(nugsize_)
	    
    {
        // instantiate the DenseGP_GPU objects
        create_emulators();
    }

    // destructor
    ~MultiOutputGP_GPU() {
        for (auto em : emulators) {
            delete em;
        }
        emulators.clear();
    }

};

#endif

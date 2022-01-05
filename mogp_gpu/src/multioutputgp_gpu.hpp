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
#include <omp.h>

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

    int get_n(void) const
    {
        return inputs.rows();
    }

    int n_emulators(void) const
    {
        return emulators.size();
    }

    int n_data_params(void) const
    {   
        if (emulators.size() > 0)
            return emulators[0]->get_theta().get_n_data();
        return 0;
    }

    int n_corr_params(void) const
    {   
        if (emulators.size() > 0)
            return emulators[0]->get_theta().get_n_corr();
        return 0;
    }

    inline nugget_type get_nugget_type(void) const { return nug_type;}

    inline REAL get_nugget_size(void) const { return nug_size;}

    void reset_fit_status(void) {
        for (unsigned int idx=0; idx < emulators.size(); ++idx) {
            emulators[idx]->reset_theta_fit_status();
        }   
    }

    void create_priors_for_emulator(unsigned int emulator_index,
                                    int n_corr,
                                    prior_type corr_dist, REAL corr_p1, REAL corr_p2,
                                    prior_type cov_dist, REAL cov_p1, REAL cov_p2,
                                    prior_type nug_dist, REAL nug_p1, REAL nug_p2)
    {   
        if (emulators.size() <= emulator_index)
            throw std::runtime_error("Invalid emulator index for setting priors");
        emulators[emulator_index]->create_gppriors(n_corr, corr_dist, corr_p1, corr_p2, 
                                                  cov_dist, cov_p1, cov_p2,
                                                  nug_dist, nug_p1, nug_p2);
    }

    std::vector<unsigned int> get_fitted_indices(void) const
    {
        std::vector<unsigned int> fitted_indices;
        for (unsigned int idx=0; idx < emulators.size(); ++idx) {
            if (emulators[idx]->get_theta_fit_status()) fitted_indices.push_back(idx);
        }
        return fitted_indices;
    }

    std::vector<unsigned int> get_unfitted_indices(void) const
    {
        std::vector<unsigned int> unfitted_indices;
        for (unsigned int idx=0; idx < emulators.size(); ++idx) {
            if ( ! emulators[idx]->get_theta_fit_status()) unfitted_indices.push_back(idx);
        }
        return unfitted_indices;
    }

    // make a single prediction per emulator(mainly for testing - most use-cases will use predict_batch or predict_deriv_batch)
    vec predict(mat_ref testing)
    {
        vec results(emulators.size());
        #pragma omp parallel for
        for (unsigned int i=0; i< emulators.size(); ++i) {
            // check whether emulator is fit OK
            if ( emulators[i]->get_theta_fit_status()) {
                results[i] = emulators[i]->predict(testing);
            }
        }
        return results;
    }

    // variance of a single prediction (mainly for testing - most use-cases will use predict_variance_batch)
    vec predict_variance(mat_ref testing, vec_ref var)
    {
        vec results(emulators.size());
        #pragma omp parallel for
        for (unsigned int i=0; i< emulators.size(); ++i) {
            if ( emulators[i]->get_theta_fit_status()) {
                results[i] = emulators[i]->predict_variance(testing, var);
            }
        }
        return results;
    }

    // Use the GP emulators to calculate a prediction on testing points, without calculating variance or derivative
    void predict_batch(mat_ref testing, mat_ref results)
    {

        #pragma omp parallel for
        for (unsigned int i=0; i< emulators.size(); ++i) {
            if ( emulators[i]->get_theta_fit_status()) {
                emulators[i]->predict_batch(testing, results.row(i));
            }
        }
        
    }

    // Use the GP emulators to calculate a prediction on testing points, also calculating variance
    void predict_variance_batch(mat_ref testing, mat_ref means, mat_ref vars)
    {
        #pragma omp parallel for
        for (unsigned int i=0; i< emulators.size(); ++i) {
            if ( emulators[i]->get_theta_fit_status()) {
                emulators[i]->predict_variance_batch(testing, means.row(i), vars.row(i));
             }
            
        }
     
    }

    // Use the GP emulator to calculate derivative of  prediction on testing points
    void predict_deriv(mat_ref testing, std::vector<mat_ref> results)
    {
        #pragma omp parallel for
        for (unsigned int i=0; i< emulators.size(); ++i) {
            if ( emulators[i]->get_theta_fit_status()) {
                emulators[i]->predict_deriv(testing, results[i]);
            }
        }
    }

    void fit(std::vector<GPParams> thetas) {
        #pragma omp parallel for
        for (unsigned int i=0; i< emulators.size(); ++i) {      
            emulators[i]->fit(thetas[i]);
        }   
    }

    void fit(mat_ref thetas) {
        #pragma omp parallel for
        for (unsigned int i=0; i< emulators.size(); ++i) {      
            emulators[i]->fit(thetas.row(i));
        }   
    }    

    void fit_emulator(unsigned int index, GPParams& theta) {
        emulators.at(index)->fit(theta);
    }

    void fit_emulator(unsigned int index, vec& theta) {
        emulators.at(index)->fit(theta);
    }    

    void create_emulators() {
        unsigned int testing_size_per_emulator = testing_size / targets.size();
        for (auto targ : targets) {
            mat dummy_design_matrix;
            // emulators will all have same starting parameters apart from targets
            emulators.push_back(new DenseGP_GPU(
                inputs, 
                targ, 
                testing_size_per_emulator, 
                dummy_design_matrix,
                kern_type,
                nug_type,
                nug_size)
            );
        }
    }

    DenseGP_GPU* get_emulator(unsigned int index) {
        return emulators.at(index);
    }

    // constructor
    MultiOutputGP_GPU(mat_ref inputs_,
	      std::vector<vec>& targets_,
	      unsigned int testing_size_,
          BaseMeanFunc* mean_=NULL,
          kernel_type kern_=SQUARED_EXPONENTIAL,
          nugget_type nugtype_=NUG_ADAPTIVE,
          double nugsize_=0.)
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

#ifndef FITTING_HPP
#define FITTING_HPP

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <ctime>
#include <random>

#include <math.h>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>
#include <omp.h>

#include "densegp_gpu.hpp"
#include "multioutputgp_gpu.hpp"

typedef dlib::matrix<double,0,1> column_vector;

class GPWrapper {
  // class that will translate dlib column vectors into eigen vectors and vice
  // versa, such that the gp's logpost and logpost_deriv methods can be used
  // with dlib.

private:
  DenseGP_GPU& gp;

public: 
// constructor
  GPWrapper(DenseGP_GPU& _gp) : gp(_gp) {}

  double logpost(column_vector theta) {
    std::vector<REAL> new_theta(theta.begin(), theta.end());
    double* ptr = &new_theta[0];
    vec theta_vec = Eigen::Map<vec>(ptr, new_theta.size());
    return gp.get_logpost(theta_vec);
  }

  column_vector logpost_deriv(column_vector theta) {
    std::vector<REAL> new_theta(theta.begin(), theta.end());
    double* ptr = &new_theta[0];
    vec theta_vec = Eigen::Map<vec>(ptr, new_theta.size());
    vec deriv(theta_vec.size());
    gp.logpost_deriv(deriv);
    std::vector<double> lpderiv(deriv.data(), deriv.data()+deriv.size());
    
    column_vector logpostderiv(lpderiv.size());
    // (is there really no better way of initializing dlib vector??)
    for (unsigned int i=0; i<lpderiv.size(); ++i) logpostderiv(i) = lpderiv[i];
    return logpostderiv;
  }

  vec theta() {
    return gp.get_theta();
  }

};



void fit_single_GP_MAP(DenseGP_GPU& gp, const int n_tries=15, const std::vector<double> theta0=std::vector<double>()) {
  // Fit the hyperparameters of a Gaussian Process by minimizing the
  // negative log-posterior.
  GPWrapper gpw(gp);

  std::vector< std::vector<double> > all_params;  
  // if we're given an initial set of theta values, put this at the start of all_params
  if (theta0.size() > 0) {
    if (theta0.size() != gp.get_n_params() ) 
      throw std::runtime_error("length of theta0 must equal n_params of GP.");
    all_params.push_back(theta0);
  }

  // generate random starting values for theta
  std::random_device rd;
  std::mt19937 e2(rd());

  std::uniform_real_distribution<> dist(-2.5, 2.5);

  
  for (int itry=all_params.size(); itry<n_tries; ++itry) {
    std::vector<REAL> v(gp.get_n_params());
    std::generate(v.begin(), v.end(), [&dist, &e2](){ return dist(e2);});
   
    all_params.push_back(v);
  }
  
  std::vector<double> minvals;
  std::vector<vec> thetavals;
  for (auto it = all_params.begin(); it != all_params.end(); it++) {
    
    double minf; // minimum objective function from optimizer
    /// copy the std::vector into dlib column_matrix
    column_vector theta((*it).size());
    for (unsigned int i=0; i<(*it).size(); ++i) theta(i) = (*it)[i];
    try {
      minf = find_min(dlib::bfgs_search_strategy(),  // Use BFGS search algorithm
             dlib::objective_delta_stop_strategy(1e-9), // Stop when the change in func value is less than 1e-7
             [&gpw](const column_vector& a) {
              return gpw.logpost(a);
            },
            [&gpw](const column_vector& b) {
              return gpw.logpost_deriv(b);
            },
            theta, -1);
      minvals.push_back(minf);
      thetavals.push_back(gpw.theta());    
    } catch(std::exception &e) {
      std::cout << "dlib optimization failed: " << e.what() << std::endl;
    }
  }
  
  if (minvals.size() == 0) {
    std::cout<<"All minimization tries failed"<<std::endl;
    gp.reset_theta_fit_status();
  } else {
      int minvalIndex = std::min_element(minvals.begin(),minvals.end()) - minvals.begin();
      vec best_theta = thetavals.at(minvalIndex);
      gp.fit(best_theta);

  }
  
  return;
}

void fit_GP_MAP(MultiOutputGP_GPU& mogp, const int n_tries=15, const std::vector<double> theta0=std::vector<double>()) {
  #pragma omp parallel for
  for (unsigned int i=0; i< mogp.n_emulators(); ++i) {
    fit_single_GP_MAP(*(mogp.get_emulator(i)), n_tries, theta0);
  }
  return;
}

#endif
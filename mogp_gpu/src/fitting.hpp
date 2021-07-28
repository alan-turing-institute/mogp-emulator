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
#include <nlopt.hpp>

#include "gp_gpu.hpp"


double objective_function(const std::vector<double> &x, std::vector<double> &grad, void* f_data) {

  DenseGP_GPU* gp = (DenseGP_GPU*)f_data;

  nugget_type nug_type = gp->get_nugget_type();
  std::vector<double> xcopy = x;
  double* ptr = &xcopy[0];
  vec theta_vec = Eigen::Map<vec>(ptr, xcopy.size());
  // get the gradients
  vec lpd(gp->get_n_params());
  gp->logpost_deriv(lpd);
  Eigen::VectorXd::Map(&grad[0], lpd.size()) = lpd;
  
  return gp->get_logpost(theta_vec);
}


void fit_GP_MAP(DenseGP_GPU& gp, int n_tries=15, std::string method="L-BFGS-B") {
  // Fit the hyperparameters of a Gaussian Process by minimizing the
  // negative log-posterior.

  nlopt::opt optimizer(nlopt::LD_LBFGS, gp.get_n_params());

  optimizer.set_min_objective(objective_function, &gp);
  
  optimizer.set_xtol_rel(1e-8);
  // generate random starting values for theta
  std::random_device rd;
  std::mt19937 e2(rd());

  std::uniform_real_distribution<> dist(-2.5, 2.5);

  std::vector< std::vector<double> > all_params;
  for (int itry=0; itry<n_tries; ++itry) {
    std::vector<REAL> v(gp.get_n_params());
    std::generate(v.begin(), v.end(), [&dist, &e2](){ return dist(e2);});
   
    all_params.push_back(v);
  }
  std::cout<<"here we go..."<<std::endl;
  std::vector<double> minvals;
  std::vector<vec> thetavals;
  for (auto it = all_params.begin(); it != all_params.end(); it++) {
    
    double minf; // minimum objective function from optimizer
    try{
      nlopt::result result = optimizer.optimize((*it), minf);
      std::cout << "found minimum at f(" << (*it)[0] << "," << (*it)[1] << ") = "
          << std::setprecision(10) << minf << std::endl;
      minvals.push_back(minf);
      thetavals.push_back(gp.get_theta());
    }
    catch(std::exception &e) {
      std::cout << "nlopt failed: " << e.what() << std::endl;
    }
    
  }
  if (minvals.size() == 0) {
    std::cout<<"All minimization tries failed"<<std::endl;
  } else {
      int minvalIndex = std::min_element(minvals.begin(),minvals.end()) - minvals.begin();
      vec best_theta = thetavals.at(minvalIndex);
      nugget_type nug_type = gp.get_nugget_type();
      gp.fit(best_theta, nug_type);
  }
  std::cout<<" theta is now "<<gp.get_theta()<<std::endl;

  //ZeroMeanFunc* meanfunc = new ZeroMeanFunc();

// DenseGP_GPU* newgp = new DenseGP_GPU(gp->get_inputs(), gp->get_targets(), 2000, meanfunc);
  //return gp;
  return;
}


#endif

#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <random>

#include <math.h>
#include <nlopt.h>

#include "types.hpp"
#include "gp_gpu.hpp"

/*
struct RandomGenerator {
  double minValue;
  double maxValue;
  RandomGenerator(double min, double max) :
    minValue(min),
    maxValue(max) {
  }
    int operator()() {
        std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> dist(minValue, maxValue);
        return rand() % maxValue
    }
};
*/


DenseGP_GPU fit_GP_MAP(DenseGP_GPU& gp, int n_tries=15, std::string method="L-BFGS-B") {
  // Fit the hyperparameters of a Gaussian Process by minimizing the
  // negative log-posterior.

  nugget_type nug_type = gp.get_nugget_type();

  // generate random starting values for theta
  std::random_device rd;
  std::mt19937 e2(rd());

  std::uniform_real_distribution<> dist(-2.5, 2.5);

  std::vector< vec > all_params;
  for (int itry=0; itry<n_tries; ++itry) {

    std::vector<REAL> v(gp.get_n_params());

    std::generate(v.begin(), v.end(), [&dist, &e2](){ return dist(e2);});

    vec theta_vec = Eigen::Map<vec, Eigen::Unaligned>(v.data(), v.size());


    all_params.push_back(theta_vec);
  }

  for (auto it = all_params.begin(); it != all_params.end(); it++) {

  }


  return gp;
}

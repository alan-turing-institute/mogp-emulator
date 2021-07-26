/// instantiate a DenseGP_GPU and fit the hyperparameters.

#include<iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <assert.h>
#include <stdexcept>

#include <math.h>
#include <nlopt.h>

#include "../src/fitting.hpp"
#include "../src/meanfunc.hpp"

typedef double REAL;


void testFit() {

  mat inputs(2,3);
  inputs << 1., 2., 3.,
            4., 5., 6.;
  vec targets(2);
  targets << 4., 6.;

  unsigned int max_batch_size = 2000;

  ZeroMeanFunc* meanfunc = new ZeroMeanFunc();

  DenseGP_GPU gp(inputs, targets, max_batch_size, meanfunc);

  vec theta = vec::Constant(gp.get_n_params(),1, -1.0);

  /// gp.fit(theta, NUG_ADAPTIVE);

  gp = fit_GP_MAP(gp);

}

int main() {
  testFit();
  return 0;
}
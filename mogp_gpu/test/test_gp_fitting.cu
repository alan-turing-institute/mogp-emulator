/// instantiate a DenseGP_GPU and fit the hyperparameters.

#include<iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <assert.h>
#include <stdexcept>

#include <math.h>

#include "../src/fitting.hpp"
#include "../src/meanfunc.hpp"

typedef double REAL;


void testSingleFit() {
  /// fit a single GP
  mat inputs(2,3);
  inputs << 1., 2., 3.,
            4., 5., 6.;
  vec targets(2);
  targets << 4., 6.;

  unsigned int max_batch_size = 2000;

  ZeroMeanFunc* meanfunc = new ZeroMeanFunc();

  DenseGP_GPU gp(inputs, targets, max_batch_size, meanfunc);

  //vec theta = vec::Constant(gp.get_n_params(),1, -1.0);

  /// gp.fit(theta, NUG_ADAPTIVE);

  fit_single_GP_MAP(gp);

}

void testMultiFit() {
  // fit a MultiOutputGP
  std::cout<<std::endl<<"Fitting Multi-Output GP"<<std::endl;
  mat inputs(2,3);

  inputs << 1., 2., 3., 
            4., 5., 6.;
  std::vector<vec> targets;
  
  vec targ0(2);
  targ0 << 4., 6. ;
  targets.push_back(targ0);
  vec targ1(2);
  targ0 << 5., 6. ;
  targets.push_back(targ1);
  
  unsigned int max_batch_size = 2000;
  // make a polynomial mean function 
  std::vector< std::pair<int, int> > dims_powers;
  dims_powers.push_back(std::make_pair<int, int>(0,1));
  dims_powers.push_back(std::make_pair<int, int>(0,2));
  PolyMeanFunc* meanfunc = new PolyMeanFunc(dims_powers);
  // instantiate the GP
  MultiOutputGP_GPU mgp(inputs, targets, max_batch_size, meanfunc);  

  fit_GP_MAP(mgp);

  // see what parameters we now have 
  DenseGP_GPU* em0 = mgp.get_emulator(0);
  std::cout<<" n params is "<<em0->get_n_params()<<std::endl;
  std::cout<<" theta for emulator 0 "<<std::endl<<em0->get_theta()<<std::endl;
  DenseGP_GPU* em1 = mgp.get_emulator(1);
  std::cout<<" theta for emulator 1 "<<std::endl<<em1->get_theta()<<std::endl;
}

int main() {
  testSingleFit();
  testMultiFit();
  return 0;
}
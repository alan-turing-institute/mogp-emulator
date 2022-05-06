/// Test instantiating and using a DenseGP_GPU object

#include<iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <assert.h>
#include <stdexcept>

#include <math.h>

#include "../src/densegp_gpu.hpp"

typedef double REAL;


void testGP() {

  mat inputs(2,3);

  inputs << 1., 2., 3.,
            4., 5., 6.;
  vec targets(2);
  targets << 4., 6.;
           
  unsigned int max_batch_size = 2000;
  ZeroMeanFunc* meanfunc = new ZeroMeanFunc();

  kernel_type kernel = SQUARED_EXPONENTIAL;
  // instantiate the GP
  DenseGP_GPU gp(inputs, targets, max_batch_size, meanfunc);
  vec theta = vec::Constant(gp.get_n_data(),1, -1.0);
  
  gp.fit(theta);
  mat x_predict(2,3);
  x_predict << 2., 3., 4.,
               7., 8., 9.;
  vec result(2);
  gp.predict_batch(x_predict, result);

  std::cout<<" result "<< result <<std::endl;

}

void test_set_params() {
  mat inputs(2,3);

  inputs << 1., 2., 3.,
            4., 5., 6.;
  vec targets(2);
  targets << 4., 6.;
         
  unsigned int max_batch_size = 2000;
  ZeroMeanFunc* meanfunc = new ZeroMeanFunc();

  kernel_type kernel = SQUARED_EXPONENTIAL;
  // instantiate the GP
  DenseGP_GPU gp(inputs, targets, max_batch_size, meanfunc);
  GPParams theta = gp.get_theta();
  assert(theta.get_nugget_type() == NUG_ADAPTIVE);
  assert(theta.get_nugget_size() == 0.0);
  gp.fit(theta);
}

int main() {
  testGP();
  test_set_params();
  return 0;
}
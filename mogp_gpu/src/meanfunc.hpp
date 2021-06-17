#ifndef MEANFUNC_HPP
#define MEANFUNC_HPP

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "types.hpp"


class BaseMeanFunc {

public:

  virtual ~BaseMeanFunc(){};

  // Performs a single evaluation of the mean function at specific inputs
  virtual vec mean_f(mat_ref xs,
		     vec_ref params) = 0;

  //Derivative wrt theta of the mean function at specific inputs
  virtual vec mean_deriv(mat_ref xs,
			 vec_ref params) = 0;


  //Derivative wrt x of the mean function at specific inputs
  virtual mat mean_inputderiv(mat_ref xs,
			      vec_ref params) = 0;

  // Return the number of parameters in the function.
  virtual int get_n_params(mat_ref xs) = 0;

};

class ZeroMeanFunc : public BaseMeanFunc {
public:

  // Zero mean function

  ZeroMeanFunc() {};

  virtual ~ZeroMeanFunc() {};

  inline virtual int get_n_params(mat_ref xs) { return 0; }


  inline virtual vec mean_f(mat_ref xs,
			    vec_ref params) {
    vec result = vec::Constant(xs.rows(),1, 0.);
    return result;
  }

  inline virtual vec mean_deriv(mat_ref xs,
				vec_ref params) {
    vec result = vec::Constant(xs.rows(),1, 0.);
    return result;
  }

  inline virtual mat mean_inputderiv(mat_ref xs,
				     vec_ref params) {
    mat result = mat::Constant(xs.cols(),xs.rows(), 0.);
    return result;
  }


};



class ConstMeanFunc : public BaseMeanFunc {
public:

  // Constant mean function

  ConstMeanFunc() {};

  virtual ~ConstMeanFunc() {};

  inline virtual int get_n_params(mat_ref xs) { return 1; }


  inline virtual vec mean_f(mat_ref xs,
			    vec_ref params) {
    vec result = vec::Constant(xs.rows(),1,params(0));
    return result;
  }

  inline virtual vec mean_deriv(mat_ref xs,
				vec_ref params) {
    vec result = vec::Constant(xs.rows(),1, 1.);
    return result;
  }

  inline virtual mat mean_inputderiv(mat_ref xs,
				     vec_ref params) {
    mat result = mat::Constant(xs.cols(),xs.rows(), 0.);
    return result;
  }


};


#endif

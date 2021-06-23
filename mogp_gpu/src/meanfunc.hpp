#ifndef MEANFUNC_HPP
#define MEANFUNC_HPP

#include <string>
#include <vector>
#include <utility>

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
  virtual mat mean_deriv(mat_ref xs,
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
    if ( params.rows() != get_n_params(xs))
      throw std::runtime_error("Expected params list of length "+ std::to_string(get_n_params(xs)));
    vec result = vec::Constant(xs.rows(),1, 0.);
    return result;
  }

  inline virtual mat mean_deriv(mat_ref xs,
				vec_ref params) {
    if ( params.rows() != get_n_params(xs))
      throw std::runtime_error("Expected params list of length "+ std::to_string(get_n_params(xs)));
    mat result = mat::Constant(xs.rows(),1, 0.);
    return result;
  }

  inline virtual mat mean_inputderiv(mat_ref xs,
				     vec_ref params) {
    if ( params.rows() != get_n_params(xs))
      throw std::runtime_error("Expected params list of length "+ std::to_string(get_n_params(xs)));
    mat result = mat::Constant(xs.cols(),xs.rows(), 0.);
    return result;
  }

};

class FixedMeanFunc : public BaseMeanFunc {

public:

  // Fixed mean function

  FixedMeanFunc(REAL value_): value(value_) {};

  virtual ~FixedMeanFunc() {} ;

  inline virtual int get_n_params(mat_ref xs) { return 0; }


  inline virtual vec mean_f(mat_ref xs,
			    vec_ref params) {
    if ( params.rows() != get_n_params(xs))
      throw std::runtime_error("Expected params list of length "+ std::to_string(get_n_params(xs)));
    vec result = vec::Constant(xs.rows(),1, value);
    return result;
  }

  inline virtual mat mean_deriv(mat_ref xs,
				vec_ref params) {
    if ( params.rows() != get_n_params(xs))
      throw std::runtime_error("Expected params list of length "+ std::to_string(get_n_params(xs)));
    mat result = mat::Constant(xs.rows(),1, 0.);
    return result;
  }

  inline virtual mat mean_inputderiv(mat_ref xs,
				     vec_ref params) {
    if ( params.rows() != get_n_params(xs))
      throw std::runtime_error("Expected params list of length "+ std::to_string(get_n_params(xs)));
    mat result = mat::Constant(xs.cols(),xs.rows(), 0.);
    return result;
  }

private:
  REAL value;

};


class ConstMeanFunc : public BaseMeanFunc {

public:

  // Constant mean function

  ConstMeanFunc() {};

  virtual ~ConstMeanFunc() {};

  inline virtual int get_n_params(mat_ref xs) { return 1; }


  inline virtual vec mean_f(mat_ref xs,
			    vec_ref params) {
    if ( params.rows() != get_n_params(xs))
      throw std::runtime_error("Expected params list of length "+ std::to_string(get_n_params(xs)));
    vec result = vec::Constant(xs.rows(),1,params(0));
    return result;
  }

  inline virtual mat mean_deriv(mat_ref xs,
				vec_ref params) {
    if ( params.rows() != get_n_params(xs))
      throw std::runtime_error("Expected params list of length "+ std::to_string(get_n_params(xs)));
    mat result = mat::Constant(xs.rows(),1, 1.);
    return result;
  }

  inline virtual mat mean_inputderiv(mat_ref xs,
				     vec_ref params) {
    if ( params.rows() != get_n_params(xs))
      throw std::runtime_error("Expected params list of length "+ std::to_string(get_n_params(xs)));
    mat result = mat::Constant(xs.cols(),xs.rows(), 0.);
    return result;
  }


};


class PolyMeanFunc : public BaseMeanFunc {

public:

  // Polynomial mean function

  // constructor takes a vector of pair<int, int>.  In each pair,
  // the first number represents the dimension index of the input, and
  // the second is the power to which that will be raised.
  // So, for example, if we have 1D input, and we want a 2nd order polynomial, we would have
  // <0,1>,<0,2>, corresponding to "theta_0 + theta_1 * x[0] + theta_2 * x[0]^2"
  // If we have 2D input, and we want a function linear in both dimensions, we would have
  // <0,1>, <1,1>, corresponding to "theta_0 + theta_1 * x[0] + theta_2 * x[1]"
  // The number of parameters is always 1 greater than the number of pairs in the vector,
  // as we allow for a const term as well as the coefficients multiplying these inputs/powers.
  // The const term will always be the first of the parameters.
  PolyMeanFunc(std::vector<std::pair<int,int> > dp) : dims_powers(dp) {};

  virtual ~PolyMeanFunc() {};

  inline virtual int get_n_params(mat_ref xs) { return dims_powers.size() + 1; }

  inline virtual vec mean_f(mat_ref xs,
			    vec_ref params) {
    if ( params.rows() != get_n_params(xs))
      throw std::runtime_error("Expected params list of length " +  std::to_string(get_n_params(xs)));
    vec result(xs.rows());
    for (unsigned int i=0; i< xs.rows(); ++i) {
      // have the const term as the first parameter
      REAL val = params(0);
      for (unsigned int j=0; j< dims_powers.size(); ++j) {
	int dim_index = dims_powers[j].first;
	int power = dims_powers[j].second;
	if (dim_index > xs.cols()-1) throw std::runtime_error("Dimension index must be less than "+ std::to_string(xs.cols()));
	val += params(j+1) * pow(xs(i,dim_index), power);
      }

      result(i) = val;
    }
    return result;
  }

  inline virtual mat mean_deriv(mat_ref xs,
				vec_ref params) {
    if ( params.rows() != get_n_params(xs))
      throw std::runtime_error("Expected params list of length " + std::to_string(get_n_params(xs)));
    mat result(params.rows(), xs.rows());
    for (unsigned int i=0; i< xs.rows(); ++i) {
      // deriv wrt the first param (the const term) is always 1.
      result(0,i) = 1.;
      // for the rest of the rows, evaluate x[index[j]]^pow[j]
      for (unsigned int j=0; j< dims_powers.size(); ++j) {
	int dim_index = dims_powers[j].first;
	int power = dims_powers[j].second;
	if (dim_index > xs.cols()-1) throw std::runtime_error("Dimension index must be less than D");
	result(j+1,i) = pow(xs(i,dim_index), power);
      }
    }
    return result;
  }

  inline virtual mat mean_inputderiv(mat_ref xs,
				     vec_ref params) {
    if ( params.rows() != get_n_params(xs))
      throw std::runtime_error("Expected params list of length " + std::to_string(get_n_params(xs)));
    mat result = mat::Constant(xs.cols(),xs.rows(), 0.);
    // loop through all inputs
    for (unsigned int i=0; i< xs.rows(); ++i) {
      // loop through all dimensions of the input
      for (unsigned int j=0; j< xs.cols(); ++j) {
	REAL val = 0.;
	// loop through all terms of the formula
	for (unsigned int k=0; k< dims_powers.size(); ++k) {
	  int dim_index = dims_powers[k].first;
	  int power = dims_powers[k].second;
	  if (dim_index > xs.cols()-1) throw std::runtime_error("Dimension index must be less than D");
	  // if this term deals with dimension j, differentiate and add to val
	  if (dim_index == j) {
	    val += params(k+1) * power * pow(xs(i,j),(power-1));
	  }
	}
	result(j,i) = val;
      }
    }
    return result;
  }

private:
  std::vector<std::pair<int, int> > dims_powers;

};


#endif

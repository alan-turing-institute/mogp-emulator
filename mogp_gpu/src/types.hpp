#ifndef TYPES_HPP
#define TYPES_HPP

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <Eigen/Dense>

// ----------------------------------------
// ------- Some useful typedefs for vectors and matrices

typedef double REAL;
typedef int obs_kind;
typedef typename Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat;
typedef typename Eigen::Ref<Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > mat_ref;
typedef typename Eigen::Matrix<REAL, Eigen::Dynamic, 1> vec;
typedef typename Eigen::Ref<Eigen::Matrix<REAL, Eigen::Dynamic, 1> > vec_ref;

/// Extract the raw pointer from a device vector
template <typename T>
T *dev_ptr(thrust::device_vector<T>& dv)
{
    return dv.data().get();
}

// ----------------------------------------
// enum to allow the python code to select the type of "nugget"
enum nugget_type {NUG_ADAPTIVE, NUG_FIT, NUG_FIXED};

// enum to allow python code to select Kernel function
enum kernel_type {SQUARED_EXPONENTIAL, MATERN52};

#endif

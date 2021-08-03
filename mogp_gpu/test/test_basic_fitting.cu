/// Example of using cuSOLVER Cholesky decomposition, based on StackOverflow
/// https://stackoverflow.com/questions/29196139/cholesky-decomposition-with-cuda

#include<iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <assert.h>
#include <stdexcept>

#include <math.h>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>
//#include <optimization.h>
//#include <global_optimization.h>

using namespace std;

typedef double REAL;

// In dlib, most of the general purpose solvers optimize functions that take a
// column vector as input and return a double.  So here we make a typedef for a
// variable length column vector of doubles.  This is the type we will use to
// represent the input to our objective functions which we will be minimizing.
typedef dlib::matrix<double,0,1> column_vector;

class RosenModel {

public:
  double rosen (const column_vector& m)
/*
    This function computes what is known as Rosenbrock's function.  It is 
    a function of two input variables and has a global minimum at (1,1).
    So when we use this function to test out the optimization algorithms
    we will see that the minimum found is indeed at the point (1,1). 
*/
  {
      const double x = m(0); 
      const double y = m(1);

    // compute Rosenbrock's function and return the result
      return 100.0*pow(y - x*x,2) + pow(1 - x,2);
  }

// This is a helper function used while optimizing the rosen() function.  
  const column_vector rosen_derivative (const column_vector& m)
/*!
    ensures
        - returns the gradient vector for the rosen function
!*/
  {
    const double x = m(0);
    const double y = m(1);

    // make us a column vector of length 2
    column_vector res(2);

    // now compute the gradient vector
    res(0) = -400*x*(y-x*x) - 2*(1-x); // derivative of rosen() with respect to x
    res(1) = 200*(y-x*x);              // derivative of rosen() with respect to y
    return res;
  }

};
/*
class rosen_model 
{
    
    //    This object is a "function model" which can be used with the
      //  find_min_trust_region() routine.  
    

public:
    typedef ::column_vector column_vector;
    typedef dlib::matrix<double> general_matrix;

    double operator() (
        const column_vector& x
    ) const { return rosen(x); }

    void get_derivative_and_hessian (
        const column_vector& x,
        column_vector& der,
        general_matrix& hess
    ) const
    {
        der = rosen_derivative(x);
        hess = rosen_hessian(x);
    }
};
*/

void testMinimization() {
// Set the starting point to (4,8).  This is the point the optimization algorithm
    // will start out from and it will move it closer and closer to the function's 
    // minimum point.   So generally you want to try and compute a good guess that is
    // somewhat near the actual optimum value.
    column_vector starting_point = {4, 8};


    RosenModel rm;

    std::cout<<" value of func at starting point is "<<rm.rosen(starting_point)<<std::endl;
    // The first example below finds the minimum of the rosen() function and uses the
    // analytical derivative computed by rosen_derivative().  Since it is very easy to
    // make a mistake while coding a function like rosen_derivative() it is a good idea
    // to compare your derivative function against a numerical approximation and see if
    // the results are similar.  If they are very different then you probably made a 
    // mistake.  So the first thing we do is compare the results at a test point: 
    
    //cout << "Difference between analytic derivative and numerical approximation of derivative: " 
      //   << dlib::length(dlib::derivative(rm.rosen)(starting_point) - rm.rosen_derivative(starting_point)) << endl;


    cout << "Find the minimum of the rosen function()" << endl;
    // Now we use the find_min() function to find the minimum point.  The first argument
    // to this routine is the search strategy we want to use.  The second argument is the 
    // stopping strategy.  Below I'm using the objective_delta_stop_strategy which just 
    // says that the search should stop when the change in the function being optimized 
    // is small enough.

    // The other arguments to find_min() are the function to be minimized, its derivative, 
    // then the starting point, and the last is an acceptable minimum value of the rosen() 
    // function.  That is, if the algorithm finds any inputs to rosen() that gives an output 
    // value <= -1 then it will stop immediately.  Usually you supply a number smaller than 
    // the actual global minimum.  So since the smallest output of the rosen function is 0 
    // we just put -1 here which effectively causes this last argument to be disregarded.

    find_min(dlib::bfgs_search_strategy(),  // Use BFGS search algorithm
             dlib::objective_delta_stop_strategy(1e-7), // Stop when the change in rosen() is less than 1e-7
             [&rm](const column_vector& a) {
              return rm.rosen(a);
            },
            [&rm](const column_vector& b) {
              return rm.rosen_derivative(b);
            },
            starting_point, -1);
    // Once the function ends the starting_point vector will contain the optimum point 
    // of (1,1).
    cout << "rosen solution:\n" << starting_point << endl;
    

}

int main() {
  testMinimization();
  return 0;
}
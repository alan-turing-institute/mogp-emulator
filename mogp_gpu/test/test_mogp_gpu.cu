#include <iostream>

#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <assert.h>
#include <stdexcept>

#include "../src/multioutputgp_gpu.hpp"
#include "../src/types.hpp"

typedef double REAL;

void testMOGP() {
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
    vec targ2(2);
    targ2 << 7., 9. ;
    targets.push_back(targ2);
    
    unsigned int max_batch_size = 2000;
    // make a polynomial mean function 
    std::vector< std::pair<int, int> > dims_powers;
    dims_powers.push_back(std::make_pair<int, int>(0,1));
    dims_powers.push_back(std::make_pair<int, int>(0,2));
    PolyMeanFunc* meanfunc = new PolyMeanFunc(dims_powers);
    // instantiate the GP
    MultiOutputGP_GPU mgp(inputs, targets, max_batch_size); //, meanfunc);

    std::cout<<"Num emulators "<<mgp.n_emulators()<<std::endl;

    
    DenseGP_GPU* em0 = mgp.get_emulator(0);
    vec theta = vec::Constant(em0->get_n_params(),1, -1.0);
    em0->fit(theta);
    DenseGP_GPU* em1 = mgp.get_emulator(1);
    em1->fit(theta);

    mat x_predict(2,3);
    x_predict << 2., 3., 4.,
                 7., 8., 9.;
    mat result(3,2);
    mgp.predict_batch(x_predict, result);
    std::cout<<"result"<<result<<std::endl;

}

void test_poly_meanfunc()
{
  // 1D case - 2nd order polynomial
  // mean f should be 2.2 + 4.4*x + 3.3*x^2
    const size_t N=5;
    vec x(5);
    x << 1.0, 2.0, 3.0, 4.0, 5.0;

    vec params(3);
    params << 4.4, 3.3, 2.2;

    std::vector< std::pair<int, int> > dims_powers;
    dims_powers.push_back(std::make_pair<int, int>(0,1));
    dims_powers.push_back(std::make_pair<int, int>(0,2));
    PolyMeanFunc* mf = new PolyMeanFunc(dims_powers);
    vec mean = mf->mean_f(x,params);
    vec deriv = mf->mean_deriv(x,params);
    vec inputderiv = mf->mean_inputderiv(x,params);

    std::cout<<" mean: ";
    for (unsigned int i=0; i<N; i++)
        std::cout << mean [i] << " ";
    std::cout << "\n deriv: ";
    for (unsigned int i=0; i<N; i++)
        std::cout << deriv [i] << " ";
    std::cout << "\n inputderiv: ";
    for (unsigned int i=0; i<N; i++)
        std::cout << inputderiv [i] << " ";
    std::cout << "\n";

    delete mf;
}

int main(void)
{
    testMOGP();
    return 0;
}

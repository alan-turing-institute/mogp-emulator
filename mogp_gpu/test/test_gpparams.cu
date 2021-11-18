#include <iostream>

#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <assert.h>
#include <math.h>
#include <stdexcept>
#include "../src/types.hpp"
#include "../src/gpparams.hpp"

typedef double REAL;

void test_corr_transforms() 
{
    std::cout<<"testing corr transforms"<<std::endl;
    vec x_raw(3);
    x_raw << 1.0, 2.0, 3.0 ;

    vec x_scaled = CorrTransform::raw_to_scaled(x_raw);
    assert (x_scaled[0] == exp(-0.5));
    assert (x_scaled[1] == exp(-1.0));
    assert (x_scaled[2] == exp(-1.5));
    vec x_raw_new = CorrTransform::scaled_to_raw(x_scaled);
    assert (x_raw_new[0] == x_raw[0]);
    assert (x_raw_new[1] == x_raw[1]);
    assert (x_raw_new[2] == x_raw[2]);
}

void test_cov_transforms() 
{
    std::cout<<"testing cov transforms"<<std::endl;
    vec x_raw(3);
    x_raw << 1.0, 2.0, 3.0 ;

    vec x_scaled = CovTransform::raw_to_scaled(x_raw);
    assert (x_scaled[0] == exp(1.0));
    assert (x_scaled[1] == exp(2.0));
    assert (x_scaled[2] == exp(3.0));
    vec x_raw_new = CovTransform::scaled_to_raw(x_scaled);
    assert (x_raw_new[0] == x_raw[0]);
    assert (x_raw_new[1] == x_raw[1]);
    assert (x_raw_new[2] == x_raw[2]);
}

void test_construct_with_nug_type()
{
    std::cout<<"construct GPParams with nug_type for nugget"<<std::endl;
    GPParams gpp(4,3,true, NUG_ADAPTIVE);
    //std::cout<<" size is "<<gpp.get_corr_raw().size()<<std::endl;
    assert (gpp.get_n_data() == 4);
    assert (gpp.get_corr_raw().size()==3);
    GPParams gpp_fit(4,3,true, NUG_FIT);
    assert (gpp_fit.get_n_data()==5);

}

void test_construct_with_float()
{
    std::cout<<"construct GPParams with float for nugget"<<std::endl;
    GPParams gpp(4,3,false, 0.5);
    assert (gpp.get_n_data() == 3);
    assert (gpp.get_corr_raw().size()==3);
    assert (gpp.get_nugget_type()==NUG_FIXED);
}

void test_eigen_block() {
    vec x(9);
    x << 1., 2., 3., 4., 5., 6., 7., 8., 9.;
    vec x2 = x.block(1,0,4,1);
    std::cout<<x2<<std::endl;
}

int main(void)
{
    test_construct_with_nug_type();
    test_construct_with_float();
    test_corr_transforms();
    test_cov_transforms();
    test_eigen_block();
    return 0;
}

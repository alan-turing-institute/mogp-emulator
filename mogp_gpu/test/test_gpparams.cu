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
    CorrTransform corr_trans;
    vec x_scaled = corr_trans.raw_to_scaled(x_raw);
    assert (x_scaled[0] == exp(-0.5));
    assert (x_scaled[1] == exp(-1.0));
    assert (x_scaled[2] == exp(-1.5));
    vec x_raw_new = corr_trans.scaled_to_raw(x_scaled);
    assert (x_raw_new[0] == x_raw[0]);
    assert (x_raw_new[1] == x_raw[1]);
    assert (x_raw_new[2] == x_raw[2]);
}

void test_cov_transforms() 
{
    std::cout<<"testing cov transforms"<<std::endl;
    vec x_raw(3);
    x_raw << 1.0, 2.0, 3.0 ;
    CovTransform cov_trans;
    vec x_scaled = cov_trans.raw_to_scaled(x_raw);
    assert (x_scaled[0] == exp(1.0));
    assert (x_scaled[1] == exp(2.0));
    assert (x_scaled[2] == exp(3.0));
    vec x_raw_new = cov_trans.scaled_to_raw(x_scaled);
    assert (x_raw_new[0] == x_raw[0]);
    assert (x_raw_new[1] == x_raw[1]);
    assert (x_raw_new[2] == x_raw[2]);
}

void test_construct_with_nug_type()
{
    std::cout<<"construct GPParams with nug_type for nugget"<<std::endl;
    GPParams gpp(4,3, NUG_ADAPTIVE);
    //std::cout<<" size is "<<gpp.get_corr_raw().size()<<std::endl;
    assert (gpp.get_n_data() == 4);
    assert (gpp.get_corr_raw().size()==3);
    GPParams gpp_fit(4,3, NUG_FIT);
    assert (gpp_fit.get_n_data()==5);

}

void test_construct_with_float()
{
    std::cout<<"construct GPParams with float for nugget"<<std::endl;
    GPParams gpp(4,3, 0.5);
    assert (gpp.get_n_mean() == 4);
    assert (gpp.get_n_data() == 4);
    assert (gpp.get_corr_raw().size()==3);
    assert (gpp.get_nugget_type()==NUG_FIXED);
}

void test_eigen_block() {
    vec x(9);
    x << 1., 2., 3., 4., 5., 6., 7., 8., 9.;
    vec x2 = x.block(1,0,4,1);
    vec x3 = x.block(4,0,4,1);
    std::cout<<"x2 "<<std::endl<<x2<<std::endl;
    std::cout<<"x3 "<<std::endl<<x3<<std::endl;
}

void test_concatenate_vecs() {
    GPParams gpp(4,3, 0.5);   
    vec mean(4);
    mean << 2., 3., 4., 5.;
    gpp.set_mean(mean);
    vec all_params(7); 
    all_params << gpp.get_mean(), gpp.get_data();
    assert (all_params.size() == 7);
    std::cout<<"all params "<<all_params<<std::endl;
}

int main(void)
{
    test_construct_with_nug_type();
    test_construct_with_float();
    test_corr_transforms();
    test_cov_transforms();
    test_eigen_block();
    test_concatenate_vecs();
    return 0;
}

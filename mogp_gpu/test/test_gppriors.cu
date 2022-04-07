#include <iostream>

#include <vector>
#include <set>
#include <algorithm>
#include <string>
#include <sstream>
#include <assert.h>
#include <math.h>
#include <stdexcept>
#include <random>
#include <boost/math/distributions/inverse_gamma.hpp>
#include <boost/math/tools/roots.hpp>

#include "../src/types.hpp"
#include "../src/util.hpp"
#include "../src/gpparams.hpp"
#include "../src/gppriors.hpp"

typedef double REAL;

void test_normal_prior() {
    NormalPrior np(2.,2.);
    assert(abs(np.logp(3.) - (-1.73709)) < 0.001);
    assert(np.dlogpdx(3.) == -0.25);
    assert(np.d2logpdx2(3.) == -0.25);
    std::cout<<"Normal Prior OK"<<std::endl;
}

void test_lognormal_prior() {
    LogNormalPrior lnp(2.,2.);
    std::cout<<" logp(3) is "<<lnp.logp(3.)<<std::endl;
    std::cout<<" dlogpdx(3) is "<<lnp.dlogpdx(3.)<<std::endl;
    std::cout<<" d2logpdx2(3) is "<<lnp.d2logpdx2(3.)<<std::endl;
    assert(abs(lnp.logp(3.) - (-2.7312)) < 0.001);
    assert(abs(lnp.dlogpdx(3.) - (-0.36712)) < 0.001);
    assert(abs(lnp.d2logpdx2(3.) - (0.09459)) < 0.001);
    std::cout<<"LogNormal Prior OK"<<std::endl;
}

void test_gamma_prior() {
    GammaPrior gamp(2.,2.);
    std::cout<<" logp(3) is "<<gamp.logp(3.)<<std::endl;
    std::cout<<" dlogpdx(3) is "<<gamp.dlogpdx(3.)<<std::endl;
    std::cout<<" d2logpdx2(3) is "<<gamp.d2logpdx2(3.)<<std::endl;
    assert(abs(gamp.logp(3.) - (-1.78768)) < 0.001);
    assert(abs(gamp.dlogpdx(3.) - (-0.16667)) < 0.001);
    assert(abs(gamp.d2logpdx2(3.) - (-0.11111)) < 0.001);
    std::cout<<"Gamma Prior OK"<<std::endl;
}

void test_invgamma_prior() {
    InvGammaPrior igamp(2.,2.);
    std::cout<<" logp(3) is "<<igamp.logp(3.)<<std::endl;
    std::cout<<" dlogpdx(3) is "<<igamp.dlogpdx(3.)<<std::endl;
    std::cout<<" d2logpdx2(3) is "<<igamp.d2logpdx2(3.)<<std::endl;
    assert(abs(igamp.logp(3.) - (-2.5762)) < 0.001);
    assert(abs(igamp.dlogpdx(3.) - (-0.77777)) < 0.001);
    assert(abs(igamp.d2logpdx2(3.) - (0.185185)) < 0.001);
    std::cout<<"InvGamma Prior OK"<<std::endl;
   // WeakPrior wp = gamp.default_prior(1.,5.);
}

/*
void test_spacings() {
    vec x(5);
    x << 1., 4., 3., 6., 3.;
    assert(max_spacing(x) == 5.);
    assert(median_spacing(x)==2.);
    vec x2(9);
    x2 << 1.2, 7.3, 4.6, 5.5, 5.7, 3.1, 6.5 ,3.0, 4.2;
    std::cout<<" spacings "<<max_spacing(x2)<<" "<<median_spacing(x2)<<std::endl;
    assert(max_spacing(x2) == 6.1);
    assert(abs(median_spacing(x2)-0.8)<0.0001);
}
*/

void test_isinstance() {
    LogNormalPrior lnp(2.,2.);
    bool is_lnp = instanceof<GammaPrior>(&lnp);
    std::cout<<" is_lnp? "<<is_lnp<<std::endl;
    
}

void test_root_finding() {
    std::cout<<" testing root finding"<<std::endl;
    boost::math::inverse_gamma_distribution<> dist(2., 2.);
    REAL cdfval = boost::math::cdf(dist, 17.);
    std::cout<<"cdfval "<<cdfval<<std::endl;
    REAL min_val = 1.;
    REAL max_val= 10.;
    REAL mode = sqrt(min_val*max_val);
    
    auto f = [mode, max_val](REAL x) {
        REAL a = exp(x);
        return boost::math::cdf(boost::math::inverse_gamma_distribution<>(a,(1. + a)*mode), max_val) - 0.995;
    };
    std::cout<<" value of f(5.) is "<<f(5.)<<std::endl;
    std::cout<<" value of f(8.) is "<<f(8.)<<std::endl;

    int get_digits = std::numeric_limits<REAL>::digits - 3;
    boost::math::tools::eps_tolerance<REAL> tol(get_digits); 
    boost::uintmax_t it = 20;
//    std::pair<REAL, REAL> r = boost::math::tools::bracket_and_solve_root( f, 5., 2., true, tol, it);
    std::pair<REAL, REAL> r = boost::math::tools::bisect(f, -10., 10., tol, it);
    REAL answer = r.first + (r.second - r.first)/2;
    std::cout<<" value of f(5.) is "<<f(5.)<<" answer is "<<answer<<std::endl;
}


int main(void)
{
    test_normal_prior();
    test_lognormal_prior();
    test_gamma_prior();
    test_invgamma_prior();
    test_root_finding();
 
    return 0;
}

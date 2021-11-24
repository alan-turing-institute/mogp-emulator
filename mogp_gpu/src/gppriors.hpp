#ifndef GPPRIORS_HPP
#define GPPRIORS_HPP

#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <boost/math/distributions/inverse_gamma.hpp>
#include "types.hpp"
#include "util.hpp"
#include "gpparams.hpp"

REAL max_spacing(vec inputs) {
    std::sort(inputs.data(), inputs.data()+inputs.size());
    return inputs[inputs.size()-1] - inputs[0];
}

REAL median_spacing(vec inputs) {
    if (inputs.size() < 2) {
        return 0.;
    }
    // convert into std::set to get unique and sorted
    std::set<REAL> s(inputs.data(), inputs.data()+inputs.size());
    // fill a vector of diffs between adjacent elements
    std::vector<REAL> diffs; 
    std::adjacent_difference(s.begin(), s.end(), 
                            std::back_inserter(diffs)
                            );
    // this vector will have an unwanted extra element at the start
    diffs.erase(diffs.begin());
    // now sort it again, so we can find the median
    std::sort(diffs.data(), diffs.data()+diffs.size());
    // finally, get the median - if we have an even number of 
    // elements, take the average of the middle two.
    if (diffs.size() % 2 == 1) {
        return diffs[diffs.size()/2];
    } else {
        return 0.5*(diffs[diffs.size()/2 -1] + diffs[diffs.size()/2]);
    }
}


class MeanPriors {

public: 
    MeanPriors(vec _mean, mat _cov)
    : mean(mean)
    , cov(_cov) {

    }

    inline int get_n_params() { return mean.size();}

    inline bool has_weak_priors() { return mean.size() == 0; }

    mat get_inv_cov() {
        /// TODO
        return cov;
    }

private:
    vec mean;
    mat cov;

};


class WeakPrior {
    public:
        inline virtual REAL logp(REAL x) {return 0.;}

        inline virtual REAL dlogpdx(REAL x) {return 0.;}

        inline virtual REAL d2logpdx2(REAL x) {return 0.;}

        virtual REAL d2logpdtheta2(REAL x, BaseTransform& transform) {
            REAL term1 = d2logpdx2(x) * pow(transform.dscaled_draw(x),2); 
            REAL term2 = dlogpdx(x) * (transform.d2scaled_draw2(x));
            return term1 + term2;
        }

        virtual REAL sample(BaseTransform& transform) {
            return sample();
        }

        virtual REAL sample() { 
            std::random_device rd;
            std::mt19937 e2(rd());
            std::uniform_real_distribution<> dist(0.,5.);
            return dist(e2) - 0.5;
        }

};

class PriorDist : public WeakPrior {
    /// Generic prior distribution
    public:
        virtual WeakPrior default_prior(REAL min, REAL max) { 
            assert(min > 0.);
            assert(max > 0.);
            assert(max > min);
            /// TODO
          //  if (instanceof<LogNormalPrior>(this)) {
           //     std::cout<<"I am LogNormal!"<<std::endl;
           // } else if (instanceof<GammaPrior>(this)) {
           //     std::cout<<"I am Gamma!"<<std::endl;
           // }

            return WeakPrior();
        }

        virtual WeakPrior default_prior_corr(WeakPrior& cls, vec inputs) {
            REAL min_val = median_spacing(inputs);
            REAL max_val = max_spacing(inputs);
            if ((min_val == 0.) || (max_val == 0.)) {
                std::cout<<"Too few unique inputs; defaulting to flat priors"<<std::endl;
                return WeakPrior();
            }
            return default_prior(min_val, max_val);
        }

};


class NormalPrior : public PriorDist {
    public:
        NormalPrior(REAL _mean, REAL _std)
        : mean(_mean)
        , std(_std) {}

        REAL logp(REAL x) {
            // computes log probability at given value
            return -0.5 * pow((x - mean)/std, 2) - log(std) -0.5* log(2*M_PI);
        }

        REAL dlogpdx(REAL x) {
            // derivative of log probability wrt scaled parameter
            return -1.0*(x - mean)/pow(std,2) ;
        }
        REAL d2logpdx2(REAL x) {
            // second derivative of log prob
            return -1.0*pow(std, -2) ;

        }
        REAL sample_x() {
            std::random_device rd;
            std::mt19937 e2(rd());
            std::normal_distribution<> dist(mean, std);
            return dist(e2);
        }

    private:
        REAL mean;
        REAL std;
};

class LogNormalPrior : public PriorDist {
    public:
        LogNormalPrior(REAL _shape, REAL _scale)
        : shape(_shape)
        , scale(_scale) {
            assert(shape > 0.);
            assert(scale > 0.);
        }

        REAL logp(REAL x) {
            // computes log probability at given value
            assert(x> 0.);
            return -0.5 * pow(log(x/scale)/shape,2) - 0.5*log(2*M_PI) - log(x) -log(shape);
        }

        REAL dlogpdx(REAL x) {
            // derivative of log probability wrt scaled parameter
            assert(x > 0.);
            return -1.0*log(x/scale)/pow(shape,2)/x - 1./x;
        }

        REAL d2logpdx2(REAL x) {
            // second derivative of log prob
            assert(x>0.);
            return (-1./pow(shape,2) + log(x/scale)/pow(shape,2) + 1.)/pow(x,2);
        }

        REAL sample_x() {
            std::random_device rd;
            std::mt19937 e2(rd());
            std::lognormal_distribution<> dist(scale, shape);
            return dist(e2);
        }

    private:
        REAL shape;
        REAL scale;
};

class GammaPrior : public PriorDist {
    public:
        GammaPrior(REAL _shape, REAL _scale)
        : shape(_shape)
        , scale(_scale) {
            assert(shape > 0.);
            assert(scale > 0.);
        }

        REAL logp(REAL x) {
            // computes log probability at given value
            assert(x> 0.);
            return (-1.0*shape*log(scale) - lgamma(shape) +
                (shape - 1.)*log(x) - x/scale);
            
        }

        REAL dlogpdx(REAL x) {
            // derivative of log probability wrt scaled parameter
            assert(x > 0.);
            return (shape - 1.)/x - 1./scale ;
        }

        REAL d2logpdx2(REAL x) {
            // second derivative of log prob
            assert(x>0.);
            return -1.0*(shape - 1.)/pow(x,2);
        }

        REAL sample_x() {
            std::random_device rd;
            std::mt19937 e2(rd());
            std::gamma_distribution<> dist(scale, shape);
            return dist(e2);
        }

    private:
        REAL shape;
        REAL scale;
};


class GPPriors {

public:
    GPPriors(vec _mean, vec _corr, REAL _cov, REAL _nug_size, nugget_type _nug_type=NUG_FIT)
    : mean(_mean)
    , corr(_corr)
    , cov(cov)
    , nug_size(_nug_size)
    , nug_type(_nug_type) {}


    inline vec get_mean() { return mean; }
private:
    vec mean;
    vec corr;
    REAL cov;
    REAL nug_size;
    nugget_type nug_type;
};


#endif
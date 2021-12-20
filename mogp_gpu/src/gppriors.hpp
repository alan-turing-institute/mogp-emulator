#ifndef GPPRIORS_HPP
#define GPPRIORS_HPP

#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <boost/math/distributions/inverse_gamma.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include "types.hpp"
#include "util.hpp"
#include "gpparams.hpp"

class MeanPriors {

public: 
    MeanPriors(vec _mean, mat _cov)
    : mean(mean)
    , cov(_cov) {}

    MeanPriors() {
        vec default_mean;
        mat default_cov;
        MeanPriors(default_mean, default_cov);
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

        virtual ~WeakPrior() {
            std::cout<<" in WeakPrior destructor "<<std::endl;
        }
        inline virtual REAL logp(REAL x) {return 0.;}

        inline virtual REAL dlogpdx(REAL x) {return 0.;}

        inline virtual REAL d2logpdx2(REAL x) {return 0.;}

        virtual REAL dlogpdtheta(REAL x, const BaseTransform& transform) {
            return dlogpdx(x)* transform.dscaled_draw(x);
        }

        virtual REAL d2logpdtheta2(REAL x, const BaseTransform& transform) {
            REAL term1 = d2logpdx2(x) * pow(transform.dscaled_draw(x),2); 
            REAL term2 = dlogpdx(x) * (transform.d2scaled_draw2(x));
            return term1 + term2;
        }

        virtual REAL sample(const BaseTransform& transform) {
            return sample();
        }

        virtual REAL sample() { 
            std::random_device rd;
            std::mt19937 e2(rd());
            std::uniform_real_distribution<> dist(0.,5.);
            return dist(e2) - 0.5;
        }

};
/// introduce intermediate pure virtual base class between WeakPrior
/// and the concrete Prior distributions, to implement the sample(transform) method
class PriorDist : public WeakPrior {
    public:

        virtual REAL sample_x() = 0;

        virtual REAL sample(const BaseTransform& transform) {
            return transform.scaled_to_raw(sample_x());
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

class InvGammaPrior : public PriorDist {
    public:
        InvGammaPrior(REAL _shape, REAL _scale)
        : shape(_shape)
        , scale(_scale) {
            assert(shape > 0.);
            assert(scale > 0.);
        }

        virtual ~InvGammaPrior() {
            std::cout<<" in InvGammaPrior descturcor"<<std::endl;
        }

        REAL logp(REAL x) {
            return (shape*log(scale) - lgamma(shape) -
                (shape + 1.)*log(x) - scale/x) ;       
        }

        REAL dlogpdx(REAL x) {
            return -1.0*(shape + 1.)/x + scale/pow(x,2);
        }

        REAL d2logpdx2(REAL x) {
            return (shape + 1)/pow(x,2) - 2.*scale/pow(x,3);
        }

        REAL sample_x() {
            std::random_device rd;
            std::mt19937 e2(rd());
            std::gamma_distribution<> dist(shape, scale);
            // inverse gamma is 1/Gamma, but multiply by scale^2 for 
            // consistency with Python version
            std::cout<<" in invgamma sample_x function "<<std::endl;
            return scale*scale/dist(e2);      
        }

    private:
        REAL shape;
        REAL scale;
};

class GPPriors {

public:
/*
    GPPriors(int n_corr_) 
    : n_corr(n_corr_) {
        std::cout<<" in GPPriors constructor with no args"<<std::endl;
        MeanPriors* default_mean = NULL;
        std::vector<WeakPrior*> default_corr;
        WeakPrior* default_cov = NULL;
        WeakPrior* default_nug = NULL;
        GPPriors(default_mean, default_corr, default_cov, default_nug, n_corr_);
    }
*/
    GPPriors(int n_corr_, nugget_type nug_type_=NUG_FIT, MeanPriors* mean_=NULL, WeakPrior* cov_=NULL, WeakPrior* nug_prior_=NULL)
    : n_corr(n_corr_)
    , nug_type(nug_type_)
    , mean_prior(mean_)
    , cov_prior(cov_)
    , nug_prior(nug_prior_) {
        std::cout<<" in GPPriors constructor with lots of args"<<std::endl;

    }


    ~GPPriors() {
        std::cout<<" in GPPriors desctructor"<<std::endl;
        if (mean_prior != NULL) delete mean_prior;
        if (cov_prior != NULL) delete cov_prior; 
        if (nug_prior != NULL) delete nug_prior;          
        corr_priors.clear();

        std::cout<<" end of GPPriors desctructor"<<std::endl;    
    }

    inline MeanPriors* get_mean() { return mean_prior; }

    inline void set_mean() { mean_prior = new MeanPriors();}

    inline void set_mean(MeanPriors* _mean) {mean_prior = _mean;}

    inline int n_mean() { return mean_prior->get_n_params();}

    inline std::vector<WeakPrior*> get_corr() { return corr_priors; }

    void set_corr(std::vector<WeakPrior*> _newcorr) {
        corr_priors = _newcorr;
        n_corr = corr_priors.size();
    }

    void set_corr() {
        for (int i=0; i< n_corr; ++i) {
            corr_priors.push_back(new WeakPrior());
        }   
    }

    inline WeakPrior* get_cov() { return cov_prior;}

    void set_cov(WeakPrior* _newcov) {
        cov_prior = _newcov;
    }

    void set_cov() {
        WeakPrior* wp = new WeakPrior();
        set_cov(wp);
    }

    inline nugget_type get_nugget_type() { return nug_type;}

    void set_nugget(WeakPrior* _newnugget) {
        if (nug_type == NUG_FIT) nug_prior = _newnugget;
    }

    void set_nugget() {
        if (nug_type == NUG_FIT) {
            WeakPrior* wp = new WeakPrior();
            set_nugget(wp);
        }
    }

    void check_theta(GPParams theta) {
        assert(n_corr == theta.get_n_corr());
        assert(nug_type == theta.get_nugget_type());
        assert(theta.get_data().size() > 0);
    }

    REAL logp(GPParams theta) {
        check_theta(theta);

        REAL logposterior = 0.;
        for (int i=0; i< corr_priors.size(); ++i) {
            logposterior += corr_priors[i]->logp(theta.get_corr()[i]);
        }
        
        logposterior += cov_prior->logp(theta.get_cov());
        
        if (nug_type == NUG_FIT) {
            logposterior += nug_prior->logp(theta.get_nugget_size());
        }


        return logposterior;
    }

    std::vector<REAL> dlogpdtheta(GPParams theta) {
        check_theta(theta);

        std::vector<REAL> partials;
        for (int i=0; i< corr_priors.size(); ++i) {
            partials.push_back(corr_priors[i]->dlogpdtheta(theta.get_corr()[i],CorrTransform()));
        }
        
        partials.push_back(cov_prior->dlogpdtheta(theta.get_cov(), CovTransform()));

        if (nug_type == NUG_FIT) {
            partials.push_back(nug_prior->dlogpdtheta(theta.get_nugget_size(), CovTransform())); 
        }
        return partials;
    }

    std::vector<REAL> d2logpdtheta2(GPParams theta) {
        check_theta(theta);

        std::vector<REAL> hessian;
        for (int i=0; i< corr_priors.size(); ++i) {
            hessian.push_back(corr_priors[i]->d2logpdtheta2(theta.get_corr()[i],CorrTransform()));
        }
            hessian.push_back(cov_prior->d2logpdtheta2(theta.get_cov(), CovTransform()));

        if (nug_type == NUG_FIT) {
            hessian.push_back(nug_prior->d2logpdtheta2(theta.get_nugget_size(), CovTransform())); 
        }
        return hessian;
    }

    std::vector<REAL> sample() {
        std::vector<REAL> samples;
        for (int i=0; i< corr_priors.size(); ++i) {
            samples.push_back(corr_priors[i]->sample(CorrTransform()));
        }
        std::cout<<" sampled from corr priors "<<std::endl;   
        samples.push_back(cov_prior->sample(CovTransform()));
        std::cout<<" sampled from cov prior "<<std::endl;
        if (nug_type == NUG_FIT) {
            std::cout<<" will try nug prior sample"<<std::endl;
            samples.push_back(nug_prior->sample(CovTransform())); 
        }
        std::cout<<" returning samples "<<samples.size()<<std::endl;
        return samples;
    }

    WeakPrior* make_prior(prior_type ptype, REAL param_1, REAL param_2) { 

        if (ptype == INVGAMMA) {
            return new InvGammaPrior(param_1, param_2); // shape and scale
        } else if (ptype == GAMMA) {
            return new GammaPrior(param_1, param_2); // shape and scale
        } else if (ptype == LOGNORMAL) {
            return new LogNormalPrior(param_1, param_2); // shape and scale
        } else if (ptype == WEAK) {
            return new WeakPrior();
        } else {
            std::cout<<"Non-weak Prior must be Inverse Gamma, Gamma, or LogNormal"<<std::endl;  
            return new WeakPrior();
        }
    }

    void create_corr_priors(prior_type ptype, REAL param_1, REAL param_2) {
        for (int i=0; i< n_corr; ++i) {
            WeakPrior* new_prior = make_prior(ptype, param_1, param_2);
            corr_priors.push_back(new_prior);
        }
    }

    void create_cov_prior(prior_type ptype, REAL param_1, REAL param_2) {
       cov_prior = make_prior(ptype, param_1, param_2);
    }

private:
    MeanPriors* mean_prior;
    std::vector<WeakPrior*> corr_priors;
    int n_corr;
    WeakPrior* cov_prior;
    WeakPrior* nug_prior;
    nugget_type nug_type;
};




#endif
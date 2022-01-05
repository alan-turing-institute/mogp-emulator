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


class WeakPrior {
    public:

        virtual ~WeakPrior() {
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
            return scale*scale/dist(e2);      
        }

    private:
        REAL shape;
        REAL scale;
};

class MeanPriors {

public: 
    MeanPriors(vec mean_, mat cov_)
    : mean(mean_)
    , cov(cov_) {
        set_prior_dists();
    }

    MeanPriors() {
        vec default_mean;
        mat default_cov;
        MeanPriors(default_mean, default_cov);
    }

    inline vec get_mean() const {return mean;}

    inline mat get_cov() const { return cov; }

    inline int get_n_params() const { return mean.size();}

    inline bool has_weak_priors() const { return mean.size() == 0; }

    void set_prior_dists(prior_type ptype=WEAK, REAL param_1=0., REAL param_2=0.) {
        
        prior_dists.clear();
        for (int i=0; i< get_n_params(); ++i ) {
            
            if (ptype == INVGAMMA) {
                prior_dists.push_back(new InvGammaPrior(param_1, param_2)); // shape and scale
            } else if (ptype == GAMMA) {
                prior_dists.push_back(new GammaPrior(param_1, param_2)); // shape and scale
            } else if (ptype == LOGNORMAL) {
                prior_dists.push_back(new LogNormalPrior(param_1, param_2)); // shape and scale
            } else {
                prior_dists.push_back(new WeakPrior());
            }
        }
    }
    
    std::vector<REAL> sample(const BaseTransform& transform) {
        std::vector<REAL> vals;
        for (int i=0; i< get_n_params(); ++i) {
            vals.push_back(prior_dists[i]->sample(transform));
        }
        return vals;
    }

    vec dm_dot_b(mat_ref dm) const {
        if (has_weak_priors()) return vec::Zero(dm.cols());
        if (dm.cols() != mean.size()) 
            throw std::runtime_error("Number of columns in design matrix doesn't match number of meanfunc parameters");
        return dm * mean;
    }

    mat get_inv_cov() const {
        if (has_weak_priors()) return mat::Zero(cov.rows(), cov.cols());
        return cov.inverse();
    }

    vec get_inv_cov_b() const {
       if (cov.size() == 0) return vec::Zero(1);
       else  return get_inv_cov() * mean;
    }

    REAL logdet_cov() const {
        if (has_weak_priors()) return 0.;
        return log(cov.determinant());
    }
    
private:
    vec mean;
    mat cov;
    std::vector<WeakPrior*> prior_dists;

};

class GPPriors {

public:

    GPPriors(int n_corr_, nugget_type nug_type_=NUG_FIT, MeanPriors* mean_=NULL, WeakPrior* cov_=NULL, WeakPrior* nug_prior_=NULL)
    : n_corr(n_corr_)
    , nug_type(nug_type_)
    , mean_prior(mean_)
    , cov_prior(cov_)
    , nug_prior(nug_prior_) {
    }


    ~GPPriors() {
        if (mean_prior != NULL) delete mean_prior;
        if (cov_prior != NULL) delete cov_prior; 
        if (nug_prior != NULL) delete nug_prior;          
        corr_priors.clear(); 
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

    void set_nugget(prior_type ptype, REAL param_1, REAL param_2) {
        if (nug_type == NUG_FIT) {
            WeakPrior* wp = make_prior(ptype, param_1, param_2);
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


    vec dlogpdtheta(GPParams theta) {

        check_theta(theta);

        vec partials(theta.get_n_data());
        //correlation length parameters
        for (int i=0; i< corr_priors.size(); ++i) {
            partials[i] = corr_priors[i]->dlogpdtheta(theta.get_corr()[i],CorrTransform());   
        }
        // covariance parameter
        partials[theta.get_n_corr()] = cov_prior->dlogpdtheta(theta.get_cov(), CovTransform());
        // nugget parameter (if fitted nugget)
        if (nug_type == NUG_FIT) {
        //    partials.push_back(nug_prior->dlogpdtheta(theta.get_nugget_size(), CovTransform()));
            partials[theta.get_n_data()-1] = nug_prior->dlogpdtheta(theta.get_nugget_size(), CovTransform());  
        }
        return partials;
    }

    vec d2logpdtheta2(GPParams theta) {

        check_theta(theta);

        vec hessian(theta.get_n_data());
        //correlation length parameters
        for (int i=0; i< corr_priors.size(); ++i) {
            hessian[i] = corr_priors[i]->d2logpdtheta2(theta.get_corr()[i],CorrTransform());
        }
        // covariance parameter
        hessian[theta.get_n_corr()] = cov_prior->d2logpdtheta2(theta.get_cov(), CovTransform());
        // nugget parameter (if fitted nugget)
        if (nug_type == NUG_FIT) {
            hessian[theta.get_n_data()-1] = nug_prior->d2logpdtheta2(theta.get_nugget_size(), CovTransform()); 
        }
        return hessian;
    }

    std::vector<REAL> sample() {
        std::cout<<" sample in GPPriors "<<std::endl;
        std::vector<REAL> samples; 
        if ( mean_prior != NULL) {
            samples = mean_prior->sample(CorrTransform());
        } 
        std::cout<<" sample in GPPriors - got mean samples "<<samples.size()<<std::endl;
        for (int i=0; i< corr_priors.size(); ++i) {
            samples.push_back(corr_priors[i]->sample(CorrTransform()));
        }  
        std::cout<<" sample in GPPriors - got corr samples "<<samples.size()<<std::endl;    
        samples.push_back(cov_prior->sample(CovTransform()));
        std::cout<<" sample in GPPriors - got cov samples "<<samples.size()<<std::endl;
        if (nug_type == NUG_FIT) {
            samples.push_back(nug_prior->sample(CovTransform())); 
        }
        std::cout<<" sample in GPPriors - got nug samples "<<samples.size()<<std::endl;
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
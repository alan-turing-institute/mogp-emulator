#ifndef GPPARAMS_HPP
#define GPPARAMS_HPP

#include <iostream>
#include <math.h>
#include "types.hpp"

/* GPParams class, and helper functions
Following the convention in GPParams.py we store the raw 
data values in the "data" vector, and provide easy access to
transformed subsets of this.
    The transformations between the raw parameters :math:`\theta` and the
    transformed ones :math:`(\beta, l, \sigma^2, \eta^2)` are as follows:
    
    1. **Mean:** No transformation; :math:`{\beta = \theta}`
    2. **Correlation:** The raw values are transformed via
       :math:`{l = \exp(-0.5\theta)}` such that the transformed values are the
       correlation length associated with the given input.
    3. **Covariance:** The raw value is transformed via
       :math:`{\sigma^2 = \exp(\theta)}` so that the transformed value is
       the covariance.
    4. **Nugget:** The raw value is transformed via
       :math:`{\eta^2 = \exp(\theta)}` so that the transformed value is
       the variance associated with the nugget noise.
*/

class CorrTransform {

public:
    static REAL raw_to_scaled(REAL r) {
        return exp(-0.5 * r);
    } 
    static REAL scaled_to_raw(REAL s) {
        return -2.0*log(s);
    }   
    static vec raw_to_scaled(vec r) {
        return Eigen::exp(-0.5*r.array());//.exp();    
    }
    static vec scaled_to_raw(vec s) {
        return -2.0*s.array().log();
    } 
    static REAL dscaled_draw(REAL s) {
        return -0.5*s;
    }
    static REAL d2scaled_draw2(REAL s) {
        return -0.25*s;
    }
    static vec dscaled_draw(vec s) {
        return -0.5*s;
    }
    static vec d2scaled_draw2(vec s) {
        return -0.25*s;
    }
};


class CovTransform {

public:

    static REAL raw_to_scaled(REAL r) {
        return exp(r);
    } 
    static REAL scaled_to_raw(REAL s) {
        return log(s);
    }
    static vec raw_to_scaled(vec r) {
        return r.array().exp();    
    }
    static vec scaled_to_raw(vec s) {
        return s.array().log();
    }
    static REAL dscaled_draw(REAL s) {
        return s;
    }
    static REAL d2scaled_draw2(REAL s) {
        return s;
    }
    static vec dscaled_draw(vec s) {
        return s;
    }
    static vec d2scaled_draw2(vec s) {
        return s;
    }
};

class GPParams {
    
public:
    // constructor, taking a string for the nugget
    GPParams(int _n_mean=0, int _n_corr=1, bool _fit_cov=true, nugget_type _nugget=NUG_FIT, REAL _nugget_size=0.0) 
    : n_mean(_n_mean)
    , n_corr(_n_corr)
    , fit_cov(_fit_cov)
    , n_data(_n_corr + int(_fit_cov) + int(_nugget==NUG_FIT))
    , nugget_size(_nugget_size)
    , nug_type(_nugget)
    , has_data(false)
    {
        data = vec::Zero(n_data);
    }
    // alternative constructor, for the fixed nugget case
    GPParams(int _n_mean, int _n_corr, bool _fit_cov, REAL _nugget) 
    : GPParams(_n_mean, _n_corr, _fit_cov, NUG_FIXED, _nugget ) {}

    // getter and setter methods

    inline int get_n_data() { return n_corr + int(fit_cov) + int(nug_type==NUG_FIT);}

    void set_mean(vec _new_mean) {
        mean = _new_mean;
        n_mean = _new_mean.size();
    }

    inline vec get_mean() { return mean; }

    inline int get_n_mean() { return n_mean; }  

    inline vec get_data() { return data; }

    void set_data(vec _new_data) {
        /// also resets Mean, covariance, nugget
        if (_new_data.size() != n_data) 
            throw std::runtime_error("New data not correct shape");
        data = _new_data;
        has_data = true;
        mean = vec::Zero(n_mean);
        if (! fit_cov) cov = 0.;
        if (nug_type == NUG_ADAPTIVE) nugget_size = 0.;
    }

    vec get_corr_raw() {
        return data.block(0, 0,n_corr, 1);
    }

    vec get_corr() {
        return CorrTransform::raw_to_scaled(get_corr_raw());
    }

    void set_corr(vec _new_corr) {
        data.block(0,0,n_corr,1) = CorrTransform::scaled_to_raw(_new_corr);
    }

    inline int get_n_corr() { return n_corr; }

    int get_cov_index() {
        // determine where in the data array the covariance parameter is
        if (! fit_cov) throw std::runtime_error("Covariance not being fit");
        if (nug_type == NUG_FIT) return n_data - 2;
        else return n_data - 1;
    }

    // transformed covariance parameter
    REAL get_cov() {
        if (fit_cov) {
            if (! has_data) return 0.;
            else return CovTransform::raw_to_scaled(data[get_cov_index()]);
        } else {
            return cov;
        }
    }

    void set_cov(REAL _cov) {
        if (fit_cov) {
            data[get_cov_index()] = CovTransform::scaled_to_raw(_cov);
        } else {
            cov = _cov;
        }
    }

    inline bool get_fit_cov() { return fit_cov; }

    inline void set_fit_cov(bool _fit_cov) { fit_cov = _fit_cov; }

    inline void set_nugget_type(nugget_type _nug_type) { nug_type = _nug_type; }

    inline nugget_type get_nugget_type() { return nug_type; }

    inline void set_nugget_size(REAL _nug_size) { nugget_size = _nug_size; }

    REAL get_nugget_size() { 
        if (nug_type != NUG_FIT) {
            return nugget_size; 
        } else {
            return CovTransform::raw_to_scaled(data[n_data-1]);
        }
    }   

    bool test_same_shape(GPParams other) {
        if (n_mean != other.get_n_mean()) return false;
        if (n_corr != other.get_n_corr()) return false;
        if (fit_cov != other.get_fit_cov()) return false;
        if (nug_type != other.get_nugget_type()) return false;
        return true;
    }

    void unset_data() { 
        data = vec::Zero(n_data);
        has_data = false;
    }

    inline bool data_has_been_set() { return has_data; }

private:
    REAL nugget_size;
    nugget_type nug_type;
    int n_mean;
    int n_corr;
    int n_data;
    bool fit_cov;
    bool has_data;
    vec data;
    vec mean;
    vec corr;
    REAL cov;
};

#endif
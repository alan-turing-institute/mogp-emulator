import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..GPParams import GPParams, CovTransform, CorrTransform
from ..Priors import GPPriors, min_spacing, max_spacing
from ..Priors import default_prior, default_prior_corr
from ..Priors import NormalPrior, LogNormalPrior, GammaPrior, InvGammaPrior
from scipy.stats import norm, gamma, invgamma, lognorm

def test_GPPrior():
    "test the GPPrior class"
    
    gpp = GPPriors(None, n_params=4, n_mean=0, nugget_type="fixed")
    
    # check indexing works
    
    assert gpp[0] is None
    
    # check iteration works
    
    out = []
    
    for p in gpp:
        out.append(p)
    
    assert out == [None, None, None, None]
    
    # check len works
    
    assert len(gpp) == 4
    
    gpp = GPPriors([NormalPrior(2., 3.), NormalPrior(2., 3.), NormalPrior(2., 3.)], n_params=3, n_mean=0,
                   nugget_type="pivot")
    
    # check indexing works
    
    assert isinstance(gpp[0], NormalPrior)
    
    # check iteration works
    
    out = []
    
    for p in gpp:
        out.append(p)
    
    assert all([isinstance(o, NormalPrior) for o in out])
    
    # check len works
    
    assert len(gpp) == 3
    
    gpp = GPPriors([NormalPrior(2., 3.), NormalPrior(2., 3.), NormalPrior(2., 3.)], n_params=4, n_mean=0,
                   nugget_type="fixed")

    # check indexing works

    assert isinstance(gpp[0], NormalPrior)

    # check iteration works

    out = []

    for p in gpp:
        out.append(p)

    assert all([isinstance(o, NormalPrior) if o is not None else o is None for o in out])

    # check len works

    assert len(gpp) == 4

def test_GPPrior_logp():
    "test the logp method of the GPPriors class"
    
    gpp = GPPriors([ NormalPrior(0., 1.), NormalPrior(2., 3.), InvGammaPrior(2., 3.), InvGammaPrior(1., 1.) ],
                    n_params=4, n_mean=1, nugget_type="fit")
                    
    theta = GPParams(n_mean=1, n_corr=1, nugget=True, data = np.zeros(4))
    
    logp = gpp.logp(theta)
    
    assert_allclose(logp, np.sum([ NormalPrior(0., 1.).logp(theta.mean), NormalPrior(2., 3.).logp(theta.corr),
                                   InvGammaPrior(2., 3.).logp(theta.cov), InvGammaPrior(1., 1.).logp(theta.nugget) ]))
                           
def test_GPPrior_dlogpdtheta():
    "test the dlogpdtheta method of the GPPriors class"
    
    gpp = GPPriors([ NormalPrior(0., 1.), NormalPrior(2., 3.), InvGammaPrior(2., 3.), InvGammaPrior(1., 1.)],
                    n_params=4, n_mean=1, nugget_type="fit")
                    
    theta = GPParams(n_mean=1, n_corr=1, nugget=True, data = np.zeros(4))
    
    partials = gpp.dlogpdtheta(theta)
    
    assert_allclose(partials,
                    [ float(NormalPrior(0., 1.).dlogpdtheta(theta.mean)),
                      float(NormalPrior(2., 3.).dlogpdtheta(theta.corr)*CorrTransform.dscaled_draw(theta.data[1])),
                      float(InvGammaPrior(2., 3.).dlogpdtheta(theta.cov)*CovTransform.dscaled_draw(theta.data[2])),
                      float(InvGammaPrior(1., 1.).dlogpdtheta(theta.nugget)*CovTransform.dscaled_draw(theta.data[3]))])
                      

def test_GPPrior_d2logpdtheta2():
    "test the dlogpdtheta method of the GPPriors class"
    
    gpp = GPPriors([ NormalPrior(0., 1.), NormalPrior(2., 3.), InvGammaPrior(2., 3.), InvGammaPrior(1., 1.)],
                    n_params=4, n_mean=1, nugget_type="fit")
                    
    theta = GPParams(n_mean=1, n_corr=1, nugget=True, data = np.zeros(4))
    
    hessian = gpp.d2logpdtheta2(theta)
    
    assert_allclose(hessian,
                    [ float(NormalPrior(0., 1.).d2logpdtheta2(theta.mean)),
                      float(NormalPrior(2., 3.).d2logpdtheta2(theta.corr)*CorrTransform.dscaled_draw(theta.data[1])**2
                            + NormalPrior(2., 3.).dlogpdtheta(theta.corr)*CorrTransform.d2scaled_draw2(theta.data[1])),
                      float(InvGammaPrior(2., 3.).d2logpdtheta2(theta.cov)*CovTransform.dscaled_draw(theta.data[2])**2
                            + InvGammaPrior(2., 3.).dlogpdtheta(theta.cov)*CovTransform.d2scaled_draw2(theta.data[2])),
                      float(InvGammaPrior(1., 1.).d2logpdtheta2(theta.nugget)*CovTransform.dscaled_draw(theta.data[3])**2
                            +InvGammaPrior(1., 1.).dlogpdtheta(theta.nugget)*CovTransform.d2scaled_draw2(theta.data[3]))])


def test_default_prior():
    "test default_prior function"
    
    dist = default_prior(1., 3.)
    
    assert isinstance(dist, InvGammaPrior)
    assert_allclose(invgamma.cdf(1., dist.shape, scale=dist.scale), 0.005)
    assert_allclose(invgamma.cdf(3., dist.shape, scale=dist.scale), 0.995)
    
    dist = default_prior(1., 14., dist="gamma")
    
    assert isinstance(dist, GammaPrior)
    assert_allclose(gamma.cdf(1., dist.shape, scale=dist.scale), 0.005)
    assert_allclose(gamma.cdf(14., dist.shape, scale=dist.scale), 0.995)
    
    dist = default_prior(1., 12., dist="lognormal")
    
    assert isinstance(dist, LogNormalPrior)
    assert_allclose(lognorm.cdf(1., dist.shape, scale=dist.scale), 0.005)
    assert_allclose(lognorm.cdf(12., dist.shape, scale=dist.scale), 0.995)
    
    with pytest.raises(ValueError):
        default_prior(1., 12., "log")
        
    with pytest.raises(AssertionError):
        default_prior(12., 1.)
        
    with pytest.raises(AssertionError):
        default_prior(-1., 2.)
        
    with pytest.raises(AssertionError):
        default_prior(1., -2.)
        
    assert default_prior(1.e-12, 1.e-11) is None

def test_min_spacing():
    "test min_spacing function"
    
    inputs = np.array([1., 2., 4.])
    
    assert_allclose(min_spacing(inputs), 1.)
    
    np.random.shuffle(inputs)
    
    assert_allclose(min_spacing(inputs), 1.)
    
    inputs = np.array([[1., 2.], [4., 5.]])
    
    assert_allclose(min_spacing(inputs), 1.)
    
    assert min_spacing(np.array([1.])) == 0.
    assert min_spacing(np.array([1., 1., 1.])) == 0.
    
def test_max_spacing():
    "text max_spacing function"

    inputs = np.array([1., 2., 4.])

    assert_allclose(max_spacing(inputs), 3.)
    
    np.random.shuffle(inputs)
    
    assert_allclose(max_spacing(inputs), 3.)
    
    inputs = np.array([[1., 2.], [4., 5.]])
    
    assert_allclose(max_spacing(inputs), 4.)
    
    assert max_spacing(np.array([1.])) == 0.
    assert max_spacing(np.array([1., 1., 1.])) == 0.

def test_default_prior_corr():
    "test default_prior_corr"
    
    dist = default_prior_corr(np.array([1., 2., 4.]))
    
    assert isinstance(dist, InvGammaPrior)
    assert_allclose(invgamma.cdf(1., dist.shape, scale=dist.scale), 0.005)
    assert_allclose(invgamma.cdf(3., dist.shape, scale=dist.scale), 0.995)
    
    dist = default_prior_corr(np.array([1., 2., 4.]), dist="gamma")
    
    assert isinstance(dist, GammaPrior)
    assert_allclose(gamma.cdf(1., dist.shape, scale=dist.scale), 0.005)
    assert_allclose(gamma.cdf(3., dist.shape, scale=dist.scale), 0.995)
    
    dist = default_prior_corr(np.array([1., 1., 2., 4.]))
    
    assert isinstance(dist, InvGammaPrior)
    assert_allclose(invgamma.cdf(1., dist.shape, scale=dist.scale), 0.005)
    assert_allclose(invgamma.cdf(3., dist.shape, scale=dist.scale), 0.995)
    
    assert default_prior_corr([1.]) is None
    assert default_prior_corr([1., 1., 1.]) is None
    assert default_prior_corr([1., 2.]) is None


@pytest.fixture
def dx():
    return 1.e-6

def test_NormalPrior(dx):
    "test the NormalPrior class"

    normprior = NormalPrior(2., 3.)

    assert_allclose(normprior.logp(0.5), np.log(norm.pdf(0.5, loc=2., scale=3.)))

    assert_allclose(normprior.dlogpdtheta(0.5),
                    (normprior.logp(0.5) - normprior.logp(0.5 - dx))/dx, atol=1.e-7, rtol=1.e-7)

    assert_allclose(normprior.d2logpdtheta2(0.5),
                    (normprior.dlogpdtheta(0.5) - normprior.dlogpdtheta(0.5 - dx))/dx, atol=1.e-7, rtol=1.e-7)
    
    with pytest.raises(AssertionError):
        NormalPrior(2., -1.)

def test_LogNormalPrior(dx):
    "test the LogNormalPrior class"

    lognormprior = LogNormalPrior(2., 3.)

    assert_allclose(lognormprior.logp(0.5), np.log(lognorm.pdf(0.5, 2., scale=3.)))

    assert_allclose(lognormprior.dlogpdtheta(0.5),
                    (lognormprior.logp(0.5) - lognormprior.logp(0.5 - dx))/dx, atol=1.e-6, rtol=1.e-6)

    assert_allclose(lognormprior.d2logpdtheta2(0.5),
                    (lognormprior.dlogpdtheta(0.5) - lognormprior.dlogpdtheta(0.5 - dx))/dx, atol=1.e-6, rtol=1.e-6)
                    
    with pytest.raises(AssertionError):
        LogNormalPrior(2., -1.)
    
    with pytest.raises(AssertionError):
        LogNormalPrior(-2., 1.)

def test_GammaPrior(dx):
    "test the GammaPrior class"

    gprior = GammaPrior(2., 3.)

    assert_allclose(gprior.logp(0.5), np.log(gamma.pdf(0.5, 2., scale=3.)))

    assert_allclose(gprior.dlogpdtheta(0.5),
                    (gprior.logp(0.5) - gprior.logp(0.5 - dx))/dx, atol=1.e-6, rtol=1.e-6)

    assert_allclose(gprior.d2logpdtheta2(0.5),
                    (gprior.dlogpdtheta(0.5) - gprior.dlogpdtheta(0.5 - dx))/dx, atol=1.e-5, rtol=1.e-6)
                    
    with pytest.raises(AssertionError):
        GammaPrior(2., -1.)
    
    with pytest.raises(AssertionError):
        GammaPrior(-2., 1.)

def test_InvGammaPrior(dx):
    "test the InvGammaPrior class"

    igprior = InvGammaPrior(2., 3.)

    assert_allclose(igprior.logp(0.5), np.log(invgamma.pdf(0.5, 2., scale=3.)))

    assert_allclose(igprior.dlogpdtheta(0.5),
                    (igprior.logp(0.5) - igprior.logp(0.5 - dx))/dx, atol=1.e-5, rtol=1.e-5)

    assert_allclose(igprior.d2logpdtheta2(0.5),
                    (igprior.dlogpdtheta(0.5) - igprior.dlogpdtheta(0.5 - dx))/dx, atol=1.e-5, rtol=1.e-5)
                    
    with pytest.raises(AssertionError):
        InvGammaPrior(2., -1.)
    
    with pytest.raises(AssertionError):
        InvGammaPrior(-2., 1.)
    
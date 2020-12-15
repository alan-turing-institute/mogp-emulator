import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..Priors import NormalPrior, GammaPrior, InvGammaPrior
from scipy.stats import norm, gamma, invgamma

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

def test_GammaPrior(dx):
    "test the GammaPrior class"

    gprior = GammaPrior(2., 3.)

    assert_allclose(gprior.logp(0.5), np.log(gamma.pdf(np.exp(0.5), 2., scale=3.)))

    assert_allclose(gprior.dlogpdtheta(0.5),
                    (gprior.logp(0.5) - gprior.logp(0.5 - dx))/dx, atol=1.e-6, rtol=1.e-6)

    assert_allclose(gprior.d2logpdtheta2(0.5),
                    (gprior.dlogpdtheta(0.5) - gprior.dlogpdtheta(0.5 - dx))/dx, atol=1.e-6, rtol=1.e-6)

def test_InvGammaPrior(dx):
    "test the InvGammaPrior class"

    igprior = InvGammaPrior(2., 3.)

    assert_allclose(igprior.logp(0.5), np.log(invgamma.pdf(np.exp(0.5), 2., scale=3.)))

    assert_allclose(igprior.dlogpdtheta(0.5),
                    (igprior.logp(0.5) - igprior.logp(0.5 - dx))/dx, atol=1.e-6, rtol=1.e-6)

    assert_allclose(igprior.d2logpdtheta2(0.5),
                    (igprior.dlogpdtheta(0.5) - igprior.dlogpdtheta(0.5 - dx))/dx, atol=1.e-6, rtol=1.e-6)

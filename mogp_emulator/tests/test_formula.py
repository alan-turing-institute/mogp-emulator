import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..formula import parse_factor_code, _is_float, inputstr_to_mean
from ..MeanFunction import ConstantMean, Coefficient, LinearMean
try:
    no_patsy = False
except ImportError:
    no_patsy = True

patsy_skip = pytest.mark.skipif(no_patsy, reason="patsy is needed to test formula parsing")

@patsy_skip
def test_mean_from_patsy_formula():
    pass

@patsy_skip
def test_mean_from_patsy_formula_failures():
    pass

@patsy_skip
def test_term_to_mean():
    pass

@patsy_skip
def test_term_to_mean_failures():
    pass

def test_code_to_mean():
    pass

def test_code_to_mean_failures():
    pass

@pytest.mark.parametrize("code,inputdict,result", [("x[1]", {}, "x[1]"),
                                                   ("inputs[0]", {}, "x[0]"),
                                                   ("a", {}, "a"),
                                                   ("a", {"a":0}, "x[0]"),
                                                   ("d", {"d":1}, "x[1]")])
def test_parse_factor_code(code, inputdict, result):
    "test the function to parse factor code"

    assert parse_factor_code(code, inputdict) == result

def test_parse_factor_code_failures():
    "test situations where parse_factor_code should fail"

    with pytest.raises(AssertionError):
        parse_factor_code(1)

def test_is_float():
    "test the _is_float function"

    assert _is_float("1")
    assert _is_float("-2.e-4")
    assert not _is_float("a")

@pytest.mark.parametrize("code,inputdict,params,resulttype,result",
                         [("2."  , {}     , np.zeros(0), ConstantMean, np.array([2., 2.])),
                          ("a"   , {}     , np.ones(1) , Coefficient , np.array([1., 1.])),
                          ("x[0]", {}     , np.zeros(0), LinearMean  , np.array([1., 4.])),
                          ("a"   , {"a":0}, np.zeros(0), LinearMean  , np.array([1., 4.]))])
def test_inputstr_to_mean(code, inputdict, params, resulttype, result):
    "test the inputstr_to_mean function"

    x = np.array([[1., 2., 3.], [4., 5., 6.]])

    mf = inputstr_to_mean(code, inputdict)

    assert isinstance(mf, resulttype)
    assert mf.get_n_params(x) == len(params)

    assert_allclose(mf.mean_f(x, params), result)

def test_inputstr_to_mean_failures():
    "test situation where inputstr_to_mean should fail"

    with pytest.raises(AssertionError):
        inputstr_to_mean(1)

    with pytest.raises(ValueError):
        inputstr_to_mean("x[0")

    with pytest.raises(ValueError):
        inputstr_to_mean("x(1]")

    with pytest.raises(ValueError):
        inputstr_to_mean("x[a]")

    with pytest.raises(AssertionError):
        inputstr_to_mean("x[-19]")

def test_tokenize_string():
    pass

def test_tokenize_string_failures():
    pass

def test_parse_tokens():
    pass

def test_parse_tokens_failures():
    pass

def test_eval_parsed_tokens():
    pass

def test_eval_parsed_tokens_failures():
    pass
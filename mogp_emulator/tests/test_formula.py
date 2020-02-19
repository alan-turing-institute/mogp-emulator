import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..formula import parse_factor_code, _is_float, inputstr_to_mean, tokenize_string
from ..formula import parse_tokens, eval_parsed_tokens
from ..MeanFunction import ConstantMean, Coefficient, LinearMean
from ..MeanFunction import MeanSum, MeanProduct, MeanPower, MeanComposite
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
                          ("a"   , {"a":0}, np.zeros(0), LinearMean  , np.array([1., 4.])),
                          ("x[1]", {}     , np.zeros(0), LinearMean  , np.array([2., 5.]))])
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

@pytest.mark.parametrize("code,result", [("a", ["a"]),
                                         ("a + b", ["a", "+", "b"]),
                                         ("a*b+ c", ["a", "*", "b", "+", "c"]),
                                         ("x[0]", ["x[0]"]),
                                         ("x[ 0 ]", ["x[0]"]),
                                         ("x**2", ["x", "^", "2"]),
                                         ("x^2", ["x", "^", "2"]),
                                         ("c(d)", ["c", "(", "d", ")"]),
                                         ("a*(b + c)**3 + x[a]", ["a", "*", "(", "b", "+", "c", ")",
                                                                  "^", "3", "+", "x[a]"])])
def test_tokenize_string(code, result):
    "test the tokenize_string function"

    assert tokenize_string(code) == result

def test_tokenize_string_failures():
    "test situations where tokenize_string should fail"

    with pytest.raises(AssertionError):
        tokenize_string(1)

    with pytest.raises(SyntaxError):
        tokenize_string("x[")

    with pytest.raises(SyntaxError):
        tokenize_string("x]")

    with pytest.raises(SyntaxError):
        tokenize_string("x[1] + x]")

    with pytest.raises(SyntaxError):
        tokenize_string("x[(0)]")

    with pytest.raises(SyntaxError):
        tokenize_string("call")

@pytest.mark.parametrize("token_list,output_list",
                         [(["a"], ["a"]),
                          (["a", "+", "b"], ["a", "b", "+"]),
                          (["a", "*", "b", "+", "c"], ["a", "b", "*", "c", "+"]),
                          (["x", "^", "2"], ["x", "2", "^"]),
                          (["c", "(", "d", ")"], ["c", "d", "call"]),
                          (["a", "*", "(", "b", "+", "c", ")", "^", "3", "+", "x[0]"],
                           ["a", "b", "c", "+", "3", "^", "*", "x[0]", "+"])])
def test_parse_tokens(token_list, output_list):
    "test the parse_tokens function"

    assert parse_tokens(token_list) == output_list

def test_parse_tokens_failures():
    "test situation where parse_tokens fails"

    with pytest.raises(AssertionError):
        parse_tokens("a")

    with pytest.raises(AssertionError):
        parse_tokens([1])

    with pytest.raises(SyntaxError):
        parse_tokens(["x" "+", "(", "a", "+", "b"])

    with pytest.raises(SyntaxError):
        parse_tokens(["x" "*", "a", "+", "b", ")"])

@pytest.mark.parametrize("stack,inputdict,params,resulttype,result",
                         [(["x[0]"],
                           {}     , np.zeros(0)       , LinearMean , np.array([1., 4.])),
                          (["a"],
                           {}     , np.ones(1),        Coefficient , np.array([1., 1.])),
                          (["a"],
                           {"a":1}, np.zeros(0)       , LinearMean , np.array([2., 5.])),
                          (["a", "b", "x[0]", "*", "+"],
                           {}     , np.array([1., 2.]), MeanSum    , np.array([3., 9.])),
                          (["a", "x[1]", "*"],
                           {}     , np.array([2.])    , MeanProduct, np.array([4., 10.])),
                          (["inputs[0]", "2", "^"],
                           {}     , np.zeros(0)       , MeanPower  , np.array([1., 16.])),
                          (["I", "inputs[0]", "2", "^", "call"],
                           {}     , np.zeros(0)       , MeanPower  , np.array([1., 16.])),
                          (["a", "b", "x[0]", "*", "+", "x[0]", "x[1]", "*", "call"],
                           {}     , np.array([1., 2.]), MeanComposite, np.array([5., 41.]))])
def test_eval_parsed_tokens(stack, inputdict, params, resulttype, result):
    "test the eval_parsed_tokens function"

    x = np.array([[1., 2., 3.], [4., 5., 6.]])

    mf = eval_parsed_tokens(stack, inputdict)

    assert isinstance(mf, resulttype)
    assert mf.get_n_params(x) == len(params)

    assert_allclose(mf.mean_f(x, params), result)

def test_eval_parsed_tokens_failures():
    "test situations where eval_parsed_tokens should fail"

    with pytest.raises(AssertionError):
        eval_parsed_tokens(1)

    with pytest.raises(AssertionError):
        eval_parsed_tokens([1])

    with pytest.raises(SyntaxError):
        eval_parsed_tokens(["a", "+"])

    with pytest.raises(SyntaxError):
        eval_parsed_tokens(["a", "I", "+"])

    with pytest.raises(SyntaxError):
        eval_parsed_tokens(["I", "a", "+"])

    with pytest.raises(SyntaxError):
        eval_parsed_tokens(["a", "b"])

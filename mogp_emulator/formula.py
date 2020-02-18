from .MeanFunction import MeanFunction, ConstantMean, LinearMean, Coefficient
try:
    from patsy import ModelDesc, Term, EvalFactor
except ImportError:
    raise ImportError("you must install patsy to convert formulas to mean functions")

def mean_from_formula(formula, inputdict={}):
    "function to create a mean function from a formula or patsy ModelDesc"

    if formula is None or (isinstance(formula, str) and formula.strip() == ""):
        return ConstantMean(0.)

    if isinstance(formula, str):
        model = ModelDesc.from_formula(formula)
    elif isinstance(formula, ModelDesc):
        model = formula

    model_terms = []

    for term in model.rhs_termlist:
        model_terms.append(term_to_mean(term, inputdict))

    mf = model_terms.pop(0)

    assert issubclass(type(mf), MeanFunction)

    for term in model_terms:
        mf += term

    assert issubclass(type(mf), MeanFunction)

    return mf

def term_to_mean(term, inputdict={}):
    "convert a patsy Term object into a MeanFunction object"

    assert isinstance(term, Term)

    # add leading coefficient and multiply by all factors

    mf = Coefficient()

    for factor in term.factors:
        mf *= factor_to_mean(factor, inputdict)

    assert issubclass(type(mf), MeanFunction)

    return mf

def factor_to_mean(factor, inputdict={}):
    "convert a patsy Factor object into a MeanFunction object"

    assert isinstance(factor, EvalFactor)

    tokens = parse_factor_code(factor.code, inputdict)

    mf = tokens_to_mean(tokens)

    assert issubclass(type(mf), MeanFunction)

    return mf

def parse_factor_code(code, inputdict={}):
    """
    turn code into tokens to be evaluated, making replacements from inputdict as appropriate

    Currently does not do any complex parsing, only replaces terms found in inputdict
    """

    assert isinstance(code, str), "formula input to mean function is not a string"

    if code[:6] == "inputs":
        newcode = "x"+code[6:]
    elif code in inputdict:
        newcode = "x[{}]".format(inputdict[code])
    else:
        newcode = code

    if not (newcode[:2] == "x[" and newcode[-1] == "]"):
        raise ValueError("bad formula input in mean function")

    return [newcode]

def tokens_to_mean(tokenlist):
    """
    convert a list of tokens into a mean function

    presently only works with single tokens representing linear terms in the input parameters
    """

    assert isinstance(tokenlist, list)

    return inputstr_to_mean(tokenlist[0])

def inputstr_to_mean(inputstr):
    "convert a string containing a single input of the form x[<index>] to a mean function"

    assert isinstance(inputstr, str), "formula input to mean function is not a string"

    if not (inputstr[:2] == "x[" and inputstr[-1] == "]"):
        raise ValueError("bad formula input in mean function")

    try:
        index = int(inputstr[2:-1])
    except ValueError:
        raise ValueError("index in parsed formula is not an integer")

    assert index >= 0, "index in formula parsing must be non-negative"

    return LinearMean(index)

def tokenize_string(string):
    "converts a string into a series of tokens for evaluation"

    assert isinstance(string, str)

    token_list = []
    accumulated = ""

    for char in string:
        if char in ["(", ")", "+", "^", " "]:
            if not accumulated == "":
                token_list.append(accumulated)
            token_list.append(char)
            accumulated = ""
        elif char == "*":
            if accumulated == "*":
                token_list.append("^")
                accumulated = ""
            elif not accumulated == "":
                token_list.append(accumulated)
                accumulated = "*"
            else:
                accumulated = "*"
        else:
            if accumulated == "*":
                token_list.append(accumulated)
                accumulated = ""
            accumulated += char

    if not accumulated == "":
        token_list.append(accumulated)

    outlist = [item for item in token_list if not item == " "]

    return outlist

def _is_float(val):
    "checks if a token can be converted into a float"
    try:
        float(val)
    except ValueError:
        return False
    return True

def parse_tokens(token_list):
    "parses a list of tokens into an RPN sequence of operations"

    assert isinstance(token_list, list), "input must be a list of strings"

    prev_op = True

    operator_stack = []
    output_list = []

    precedence = {"+": 2, "*": 3, "^": 4}
    l_assoc = {"+": True, "*": True, "^": False}

    for token in token_list:
        assert isinstance(token, str), "input must be a list of strings"
        if not token in ["(", ")", "+", "*", "^"]:
            output_list.append(token)
            prev_op = False
        if token == "(" and not prev_op:
            # function call, put call on the stack
            operator_stack.append("call")
            prev_op = True
        if token in ["+", "*", "^"]:
            while (len(operator_stack) >= 1 and
                   (not operator_stack[-1] == "(") and
                   (operator_stack[-1] == "call" or
                    precedence[operator_stack[-1]] > precedence[token] or
                    (precedence[operator_stack[-1]] == precedence[token] and l_assoc[token]))):
                    output_list.append(operator_stack.pop())
            operator_stack.append(token)
            prev_op = True
        if token == "(":
            operator_stack.append(token)
            prev_op = True
        if token == ")":
            while not operator_stack[-1] == "(":
                output_list.append(operator_stack.pop())
                if len(operator_stack) == 0:
                    raise SyntaxError("string expression has mismatched parentheses")
            if operator_stack[-1] == "(":
                operator_stack.pop()
            prev_op = True

    while len(operator_stack) > 0:
        operator = operator_stack.pop()
        if operator == ")" or operator == "(":
            raise SyntaxError("string expression has mismatched parentheses")
        output_list.append(operator)

    return output_list



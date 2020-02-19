from . import MeanFunction
try:
    from patsy import ModelDesc, Term, EvalFactor
    no_patsy = False
except ImportError:
    no_patsy = True

def mean_from_patsy_formula(formula, inputdict={}):
    "use patsy to parse formula before evaluating terms"

    assert not no_patsy, "patsy must be installed to parse formulas using patsy"

    if isinstance(formula, str):
        model = ModelDesc.from_formula(formula)
    elif isinstance(formula, ModelDesc):
        model = formula

    model_terms = []

    for term in model.rhs_termlist:
        model_terms.append(term_to_mean(term, inputdict))

    mf = model_terms.pop(0)

    assert issubclass(type(mf), MeanFunction.MeanFunction)

    for term in model_terms:
        mf += term

    assert issubclass(type(mf), MeanFunction.MeanFunction)

    return mf

def mean_from_string(inputstr, inputdict={}):
    "convert a string formula into a MeanFunction object"

    assert isinstance(inputstr, str)

    tokens = tokenize_string(inputstr)
    eval_stack = parse_tokens(tokens)

    mf = eval_parsed_tokens(eval_stack, inputdict)

    assert issubclass(type(mf), MeanFunction.MeanFunction)

    return mf

def term_to_mean(term, inputdict={}):
    "convert a patsy Term object into a MeanFunction object"

    assert isinstance(term, Term)

    # add leading coefficient and multiply by all factors

    mf = MeanFunction.Coefficient()

    for factor in term.factors:
        assert isinstance(factor, EvalFactor)
        mf *= mean_from_string(factor.code, inputdict)

    assert issubclass(type(mf), MeanFunction.MeanFunction)

    return mf

def parse_factor_code(code, inputdict={}):
    """
    replaces given string with aliases from inputdict and converts inputs -> x
    """

    assert isinstance(code, str), "formula input to mean function is not a string"

    if code[:6] == "inputs":
        newcode = "x"+code[6:]
    elif code in inputdict:
        newcode = "x[{}]".format(inputdict[code])
    else:
        newcode = code

    return newcode

def _is_float(val):
    "checks if a token can be converted into a float"
    try:
        float(val)
    except ValueError:
        return False
    return True

def inputstr_to_mean(inputstr, inputdict={}):
    """
    convert a string to a mean function

    makes substitutions found in inputdict to map variable names to indices
    converts numeric tokens to constant means
    any other strings not in inputdict are assumed to represent coefficients
    """

    assert isinstance(inputstr, str), "formula input to mean function is not a string"

    if _is_float(inputstr):
        return MeanFunction.ConstantMean(float(inputstr))

    inputstr = parse_factor_code(inputstr, inputdict)

    if not inputstr[0] == "x":
        return MeanFunction.Coefficient()

    if not (inputstr[:2] == "x[" and inputstr[-1] == "]"):
        raise ValueError("bad formula input in mean function")

    try:
        index = int(inputstr[2:-1])
    except ValueError:
        raise ValueError("index in parsed formula is not an integer")

    assert index >= 0, "index in formula parsing must be non-negative"

    return MeanFunction.LinearMean(index)

def tokenize_string(string):
    "converts a string into a series of tokens for evaluation"

    assert isinstance(string, str)

    token_list = []
    accumulated = ""

    for char in string:
        if char in ["(", ")", "+", "^", " ", "[", "]"]:
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

    outlist = []

    for item in token_list:
        if not item in [" ", "[", "]"]:
            outlist.append(item)
        elif item == "[":
            outlist.append(outlist.pop()+item)
        elif item == "]":
            outlist.append(outlist.pop(-2)+outlist.pop()+item)

    for item in outlist:
        if (not "[" in item and "]" in item) or (not "]" in item and "[" in item):
            raise SyntaxError("cannot nest operators in square brackets in formula input")

    return outlist

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

def eval_parsed_tokens(token_list, inputdict={}):
    "evaluate parsed tokens into a mean function"

    assert isinstance(token_list, list), "input must be a list of strings"

    op_list = ["+", "*", "^", "call"]

    stack = []

    for token in token_list:
        assert isinstance(token, str), "tokens must be strings"
        if token not in op_list:
            if token == "I":
                mf = "I"
            else:
                mf = inputstr_to_mean(token, inputdict)
            stack.append(mf)
        else:
            if len(stack) < 2:
                raise SyntaxError("string expression is not a valid mathematical expression")

            op_2 = stack.pop()
            if op_2 == "I":
                raise SyntaxError("identity operator can only be called as a function")
            assert issubclass(type(op_2), MeanFunction)
            op_1 = stack.pop()
            if token == "call":
                assert op_1 == "I" or issubclass(type(op_1), MeanFunction)
            else:
                if op_2 == "I":
                    raise SyntaxError("identity operator can only be called as a function")
                assert issubclass(type(op_1), MeanFunction), "expression was not c"

            if token == "+":
                stack.append(op_1.__add__(op_2))
            elif token == "*":
                stack.append(op_1.__mul__(op_2))
            elif token == "^":
                stack.append(op_1.__pow__(op_2))
            elif token == "call":
                if op_1 == "I":
                    stack.append(op_2)
                else:
                    stack.append(op_1.__call__(op_2))
            else:
                raise SyntaxError("string expression is not a valid mathematical expression")

    if not len(stack) == 1:
        raise SyntaxError("string expression is not a valid mathematical expression")

    return stack[0]


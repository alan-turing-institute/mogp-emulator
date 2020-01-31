import numpy as np
from functools import partial

class MeanFunction(object):
    def _check_inputs(self, x, params):
        "check the shape of the inputs"

        if len(x.shape) == 1:
            x = np.reshape(x, (-1, 1))
        assert len(x.shape) == 2, "inputs must be a 1D or 2D array"

        assert len(params.shape) == 1, "params must be a 1D array"

        assert len(params) == self.get_n_params(x), "bad length for params"

        return x, params

    def get_n_params(self, x):
        "determine the number of parameters, possibly which is dependent on x"

        raise NotImplementedError("base mean function does not implement a particular function")

    def mean_f(self, x, params):
        "returns value of mean function"

        raise NotImplementedError("base mean function does not implement a particular function")

    def mean_deriv(self, x, params):
        "returns derivative of mean function wrt params"

        raise NotImplementedError("base mean function does not implement a particular function")

    def mean_hessian(self, x, params):
        "returns hessian of mean function wrt params"

        raise NotImplementedError("base mean function does not implement a particular function")

    def mean_inputderiv(self, x, params):
        "returns derivative of mean function wrt inputs"

        raise NotImplementedError("base mean function does not implement a particular function")

    def __add__(self, other):
        "combines two mean functions"

        if issubclass(type(other), MeanFunction):
            return MeanSum(self, other)
        elif isinstance(other, (float, int)):
            return MeanSum(self, FixedMean(other))
        else:
            raise TypeError("other function cannot be added with a MeanFunction")

    def __radd__(self, other):
        "combines two mean functions"

        if issubclass(type(other), MeanFunction):
            return MeanSum(other, self)
        elif isinstance(other, (float, int)):
            return MeanSum(FixedMean(other), self)
        else:
            raise TypeError("other function cannot be added with a MeanFunction")

    def __mul__(self, other):
        "multiplies two mean functions"

        if issubclass(type(other), MeanFunction):
            return MeanProduct(self, other)
        elif isinstance(other, (float, int)):
            return MeanProduct(self, FixedMean(other))
        else:
            raise TypeError("other function cannot be added with a MeanFunction")

    def __rmul__(self, other):
        "multiplies two mean functions"

        if issubclass(type(other), MeanFunction):
            return MeanProduct(other, self)
        elif isinstance(other, (float, int)):
            return MeanProduct(FixedMean(other), self)
        else:
            raise TypeError("other function cannot be added with a MeanFunction")

class MeanSum(MeanFunction):
    def __init__(self, f1, f2):

        if not issubclass(type(f1), MeanFunction):
            raise TypeError("inputs to MeanSum must be subclasses of MeanFunction")
        if not issubclass(type(f2), MeanFunction):
            raise TypeError("inputs to MeanSum must be subclasses of MeanFunction")

        self.f1 = f1
        self.f2 = f2

    def get_n_params(self, x):
        return self.f1.get_n_params(x) + self.f2.get_n_params(x)

    def mean_f(self, x, params):
        "returns value of mean function"

        switch = self.f1.get_n_params(x)

        return (self.f1.mean_f(x, params[:switch]) +
                self.f2.mean_f(x, params[switch:]))

    def mean_deriv(self, x, params):
        "returns derivative of mean function wrt params"

        switch = self.f1.get_n_params(x)

        deriv = np.zeros((self.get_n_params(x), x.shape[0]))

        deriv[:switch] = self.f1.mean_deriv(x, params[:switch])
        deriv[switch:] = self.f2.mean_deriv(x, params[switch:])

        return deriv

    def mean_hessian(self, x, params):
        "returns hessian of mean function wrt params"

        switch = self.f1.get_n_params(x)

        hess = np.zeros((self.get_n_params(x), self.get_n_params(x), x.shape[0]))

        hess[:switch, :switch] = self.f1.mean_hessian(x, params[:switch])
        hess[switch:, switch:] = self.f2.mean_hessian(x, params[switch:])

        return hess

    def mean_inputderiv(self, x, params):
        "returns derivative of mean function wrt inputs"

        switch = self.f1.get_n_params(x)

        return (self.f1.mean_inputderiv(x, params[:switch]) +
                self.f2.mean_inputderiv(x, params[switch:]))

class MeanProduct(MeanFunction):
    def __init__(self, f1, f2):

        if not issubclass(type(f1), MeanFunction):
            raise TypeError("inputs to MeanProduct must be subclasses of MeanFunction")
        if not issubclass(type(f2), MeanFunction):
            raise TypeError("inputs to MeanProduct must be subclasses of MeanFunction")

        self.f1 = f1
        self.f2 = f2

    def get_n_params(self, x):
        return self.f1.get_n_params(x) + self.f2.get_n_params(x)

    def mean_f(self, x, params):
        "returns value of mean function"

        switch = self.f1.get_n_params(x)

        return (self.f1.mean_f(x, params[:switch])*
                self.f2.mean_f(x, params[switch:]))

    def mean_deriv(self, x, params):
        "returns derivative of mean function wrt params"

        switch = self.f1.get_n_params(x)

        deriv = np.zeros((self.get_n_params(x), x.shape[0]))

        deriv[:switch] = (self.f1.mean_deriv(x, params[:switch])*
                          self.f2.mean_f(x, params[switch:]))

        deriv[switch:] = (self.f1.mean_f(x, params[:switch])*
                          self.f2.mean_deriv(x, params[switch:]))

        return deriv

    def mean_hessian(self, x, params):
        "returns hessian of mean function wrt params"

        switch = self.f1.get_n_params(x)

        hess = np.zeros((self.get_n_params(x), self.get_n_params(x), x.shape[0]))

        hess[:switch, :switch] = (self.f1.mean_hessian(x, params[:switch])*
                                  self.f2.mean_f(x, params[switch:]))
        hess[:switch, switch:, :] = (self.f1.mean_deriv(x, params[:switch])[:,np.newaxis,:]*
                                     self.f2.mean_deriv(x, params[switch:])[np.newaxis,:,:])
        hess[switch:, :switch, :] = np.transpose(hess[:switch, switch:, :], (1, 0, 2))
        hess[switch:, switch:] = (self.f1.mean_f(x, params[:switch])*
                                  self.f2.mean_hessian(x, params[switch:]))

        return hess


    def mean_inputderiv(self, x, params):
        "returns derivative of mean function wrt inputs"

        switch = self.f1.get_n_params(x)

        return (self.f1.mean_inputderiv(x, params[:switch])*
                self.f2.mean_f(x, params[switch:]) +
                self.f1.mean_f(x, params[:switch])*
                self.f2.mean_inputderiv(x, params[switch:]))

def const_f(x, val):
    return np.broadcast_to(val, x.shape[0])

def zero_inputderiv(x):
    return np.zeros((x.shape[1], x.shape[0]))

class FixedMean(MeanFunction):
    "a mean function with a fixed function/derivative and no fitting parameters"
    def __init__(self, *args):

        if len(args) == 1 and isinstance(args[0], (float, int)):
            val = float(args[0])
            f = partial(const_f, val=val)
            deriv = zero_inputderiv
        elif len(args) == 1 or len(args) == 2:
            f = args[0]
            if len(args) == 1:
                deriv = None
            else:
                deriv = args[1]
        else:
            raise ValueError("Bad length of arguments provided to FixedMean")

        assert callable(f), "fixed mean function must be a callable function"
        if not deriv is None:
            assert callable(deriv), "mean function derivative must be a callable function"

        self.f = f
        self.deriv = deriv

    def get_n_params(self, x):
        return 0

    def mean_f(self, x, params):

        x, params = self._check_inputs(x, params)

        return self.f(x)

    def mean_deriv(self, x, params):

        x, params = self._check_inputs(x, params)

        return np.zeros((self.get_n_params(x), x.shape[0]))

    def mean_hessian(self, x, params):

        x, params = self._check_inputs(x, params)

        return np.zeros((self.get_n_params(x), self.get_n_params(x), x.shape[0]))

    def mean_inputderiv(self, x, params):

        x, params = self._check_inputs(x, params)

        if self.deriv is None:
            raise NotImplementedError("Derivative function was not provided with this FixedMean")
        else:
            return self.deriv(x)

class ConstantMean(MeanFunction):
    def get_n_params(self, x):
        return 1

    def mean_f(self, x, params):

        x, params = self._check_inputs(x, params)

        return np.broadcast_to([params[0]], x.shape[0])

    def mean_deriv(self, x, params):

        x, params = self._check_inputs(x, params)

        return np.ones((self.get_n_params(x), x.shape[0]))

    def mean_hessian(self, x, params):

        x, params = self._check_inputs(x, params)

        return np.zeros((self.get_n_params(x), self.get_n_params(x), x.shape[0]))

    def mean_inputderiv(self, x, params):

        x, params = self._check_inputs(x, params)

        return np.zeros((x.shape[1], x.shape[0]))

class LinearMean(MeanFunction):
    def get_n_params(self, x):
        return x.shape[1]

    def mean_f(self, x, params):

        x, params = self._check_inputs(x, params)

        return np.sum(params*x, axis=1)

    def mean_deriv(self, x, params):

        x, params = self._check_inputs(x, params)

        return np.transpose(x)

    def mean_hessian(self, x, params):

        x, params = self._check_inputs(x, params)

        return np.zeros((self.get_n_params(x), self.get_n_params(x), x.shape[0]))

    def mean_inputderiv(self, x, params):

        x, params = self._check_inputs(x, params)

        return np.broadcast_to(np.reshape(params, (-1, 1)), (x.shape[1], x.shape[0]))

class PolynomialMean(MeanFunction):
    def __init__(self, degree):
        assert int(degree) >= 0., "degree must be a positive integer"

        self.degree = int(degree)

    def get_n_params(self, x):
        return x.shape[1]*self.degree + 1

    def mean_f(self, x, params):
        x, params = self._check_inputs(x, params)

        n_params = self.get_n_params(x)

        indices = np.arange(0, n_params - 1) % x.shape[1]
        expon = np.arange(0, n_params - 1) // x.shape[1] + 1

        output = params[0] + np.sum(params[1:]*x[:, indices]**expon, axis = 1)

        return output

    def mean_deriv(self, x, params):

        x, params = self._check_inputs(x, params)

        n_params = self.get_n_params(x)

        deriv = np.zeros((n_params, x.shape[0]))
        deriv[0] = np.ones(x.shape[0])

        indices = np.arange(0, n_params - 1) % x.shape[1]
        expon = np.arange(0, n_params - 1) // x.shape[1] + 1

        deriv[1:,:] = np.transpose(x[:, indices]**expon)

        return deriv

    def mean_hessian(self, x, params):

        x, params = self._check_inputs(x, params)

        n_params = self.get_n_params(x)

        hess = np.zeros((n_params, n_params, x.shape[0]))

        return hess

    def mean_inputderiv(self, x, params):
        x, params = self._check_inputs(x, params)

        expon = np.reshape(np.arange(0, x.shape[0]*x.shape[1]*self.degree)//x.shape[1]//self.degree,
                           (self.degree, x.shape[0]*x.shape[1]))
        x_indices = np.reshape(np.arange(0, x.shape[0]*x.shape[1]*self.degree) % (x.shape[0]*x.shape[1]),
                               (self.degree, x.shape[0]*x.shape[1]))
        param_indices = np.reshape(np.arange(0, x.shape[0]*x.shape[1]*self.degree) % x.shape[1],
                                   (self.degree, x.shape[0]*x.shape[1])) + expon*x.shape[1]
        param_indices = np.reshape(param_indices, (self.degree, x.shape[0]*x.shape[1]))

        output = np.sum((expon + 1.)*params[1:][param_indices]*x.flatten()[x_indices]**expon, axis=0)

        return np.transpose(np.reshape(output, (x.shape[0], x.shape[1])))

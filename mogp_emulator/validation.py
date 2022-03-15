import numpy as np
from mogp_emulator.GaussianProcess import GaussianProcessBase
from mogp_emulator.MultiOutputGP import MultiOutputGP
from mogp_emulator.linalg import cholesky_factor
from scipy.stats import f

def mahalanobis(gp, valid_inputs, valid_targets, scaled=False):
    """Compute the Mahalanobis distance on a validation dataset
    
    Given a fit GP and a set of inputs and targets for validation,
    compute the Mahalanobis distance (the correlated equivalent
    of the sum of the squared standard errors):
    
    .. math::
        M = (y_{valid} - y_{pred})^T K^{-1} (y_{valid} - y_{pred})
    
    The Mahalanobis distance is expected to follow a scaled 
    Fisher-Snedecor distribution with ``(n_valid, n - n_mean - 2)``
    degrees of freedom. If ``scaled=True`` is selected, then
    the returned distance will be scaled by subtracting the
    expected mean and dividing by the standard deviation of
    this distribution. Note that the Fisher-Snedecor distribution
    is not symmetric, so this cannot be interpreted in the
    same way as standard errors, but this can nevertheless be
    a useful heuristic. By default, the Mahalanobis distance
    is not scaled, and a convenience function
    ``generate_mahal_dist`` is provided to simplify comparison
    of the Mahalanobis distance to the expected distribution.
    
    ``gp`` must be a fit ``GaussianProcess`` or ``MultiOutputGP``
    object, ``valid_inputs`` must be valid input data to the
    GP/MOGP, and ``valid_targets`` must be valid target data of
    the appropraite shape for the GP/MOGP.
    
    :param gp: A fit ``GaussianProcess`` or ``MultiOutputGP``
               object. If the GP/MOGP has not been fit, a
               ``ValueError`` will be raised.
    :type gp: ``GaussianProcess`` or ``MultiOutputGP``
    :param valid_inputs: Input points at which the GP will
                         be validated. Must correspond to
                         the appropriate inputs to the
                         provided GP.
    :type valid_inputs: ndarray
    :param valid_targets: Target points at which the GP will
                          be validated. Must correspond to
                          the appropriate target shape for
                          the provided GP.
    :type valid_targets: ndarray
    :param scaled: Flag indicating if the output Mahalanobis
                   distance should be scaled by subtracting
                   the mean and dividing by the standard
                   deviation of the expected Fisher-Snedecor
                   distribution. Optional, default is
                   ``False``.
    :type scaled: bool
    :returns: Mahalanobis distance computed based on the GP
              predictions on the validation data. If a
              multiple outputs are used, then returns a
              numpy array holding the Mahalanobis distance
              for each target.
    :rtype: float or ndarray
    """
    mean, cov, _ = gp.predict(valid_inputs, full_cov=True)
    
    if isinstance(gp, GaussianProcessBase):
        mean = [mean]
        cov = [cov]
    
    M = []

    for target, meanval, covval in zip(valid_targets, mean, cov):
        cov_inv, _ = cholesky_factor(covval, 0., "fixed")
        M.append(np.dot(target - meanval, cov_inv.solve(target - meanval)))
    
    M = np.array(M)
    
    if len(M) == 1:
        M = M[0]
    
    return M

def generate_mahal_dist(gp, valid_inputs):
    """Generate the Expected Distribution for the Mahalanobis Distance
    
    Convenience function for generating a ``scipy.stats.f`` object
    appropriate for the expected Mahalanobis distribution. If a
    ``MultiOutputGP`` object is provided, then a list of distributions
    will be returned. In all cases, the parameters will be "frozen"
    as appropriate for the data.
    
    :param gp: A fit ``GaussianProcess`` or ``MultiOutputGP``
               object.
    :type gp: ``GaussianProcess`` or ``MultiOutputGP``
    :param valid_inputs: Input points at which the GP will
                         be validated. Must correspond to
                         the appropriate inputs to the
                         provided GP.
    :type valid_inputs: ndarray
    :returns: ``scipy.stats`` distribution or list of distributions.
    :rtype: scipy.stats.rv_continuous or list
    """
    
    if isinstance(gp, GaussianProcessBase):
        emulators = [gp]
    elif isinstance(gp, MultOutputGP):
        emulators = gp.emulators
    else:
        raise TypeError("Provided GP is not a GaussianProcess or MultiOutputGP")
    
    n_valid = len(gp._process_inputs(valid_inputs))
        
    outdists = []
    for em in emulators:
        outdists.append(f(dfn=n_valid, dfd=em.n-em.n_mean-2, scale=n_valid))
    
    if len(outdists) == 1:
        outdists = outdists[1]
    
    return outdists

def standard_errors(gp, valid_inputs, valid_targets):
    """Compute standard errors on a validation dataset
    
    Given a fit GP and a set of inputs and targets for validation,
    compute the standard errors (number of standard devations
    between the true and predicted values). Numbers are left
    signed to designate the direction of the discrepancy
    (positive values indicate the emulator predictions are
    larger than the true values).
    
    ``gp`` must be a fit ``GaussianProcess`` or ``MultiOutputGP``
    object, ``valid_inputs`` must be valid input data to the
    GP/MOGP, and ``valid_targets`` must be valid target data of
    the appropraite shape for the GP/MOGP.
    
    Returns a numpy array. If a GP is provided, shape will be
    ``(n_valid,)``, while if a MOGP is provided, shape
    will be ``(n_emulators, n_valid)``.
    
    :param gp: A fit ``GaussianProcess`` or ``MultiOutputGP``
               object. If the GP/MOGP has not been fit, a
               ``ValueError`` will be raised.
    :type gp: ``GaussianProcess`` or ``MultiOutputGP``
    :param valid_inputs: Input points at which the GP will
                         be validated. Must correspond to
                         the appropriate inputs to the
                         provided GP.
    :type valid_inputs: ndarray
    :param valid_targets: Target points at which the GP will
                          be validated. Must correspond to
                          the appropriate target shape for
                          the provided GP.
    :type valid_targets: ndarray
    :returns: Numpy array holding the standard errors. If a
              ``GaussianProcess`` is provided, the result
              will have shape ``(n_valid,)`` where ``n_valid``
              is the length of the validation data. If a
              ``MultiOutputGP`` is provided, the result will
              have shape ``(n_emulators, n_valid)``.
    :rtype: ndarray
    """
    mean, cov, _ = gp.predict(valid_inputs)

    return (mean - valid_targets)/np.sqrt(cov)

def pivoted_errors(gp, valid_inputs, valid_targets, undo_pivot=True):
    """Compute correlated errors on a validation dataset
    
    Given a fit GP and a set of inputs and targets for validation,
    compute the correlated errors (number of standard devations
    between the true and predicted values, conditional on the
    errors in decreasing order). Note that because the errors are
    conditional, order matters and thus the errors are treated
    with respect to the largest one. By default, the errors
    are returned to the original ordering of the inputs (note
    that this means the)
    
    ``gp`` must be a fit ``GaussianProcess`` or ``MultiOutputGP``
    object, ``valid_inputs`` must be valid input data to the
    GP/MOGP, and ``valid_targets`` must be valid target data of
    the appropraite shape for the GP/MOGP.
    
    Returns a numpy array. If a GP is provided, shape will be
    ``(n_valid,)``, while if a MOGP is provided, shape
    will be ``(n_emulators, n_valid)``.
    
    :param gp: A fit ``GaussianProcess`` or ``MultiOutputGP``
               object. If the GP/MOGP has not been fit, a
               ``ValueError`` will be raised.
    :type gp: ``GaussianProcess`` or ``MultiOutputGP``
    :param valid_inputs: Input points at which the GP will
                         be validated. Must correspond to
                         the appropriate inputs to the
                         provided GP.
    :type valid_inputs: ndarray
    :param valid_targets: Target points at which the GP will
                          be validated. Must correspond to
                          the appropriate target shape for
                          the provided GP.
    :type valid_targets: ndarray
    :param undo_pivot: Flag indicating if the ordering of
                       the errors should be returned to the
                       original ordering. Optional, default
                       is ``True``.
    :type undo_pivot: bool
    :returns: Numpy array holding the standard errors. If a
              ``GaussianProcess`` is provided, the result
              will have shape ``(n_valid,)`` where ``n_valid``
              is the length of the validation data. If a
              ``MultiOutputGP`` is provided, the result will
              have shape ``(n_emulators, n_valid)``.
    :rtype: ndarray
    """
    mean, cov, _ = gp.predict(valid_inputs, full_cov=True)
    
    if isinstance(gp, GaussianProcessBase):
        mean = [mean]
        cov = [cov]
    
    errors = []

    for target, meanval, covval in zip(valid_targets, mean, cov):
        cov_inv, _ = cholesky_factor(covval, 0., "pivot")
        if undo_pivot:
            P = cov_inv.P
        else:
            P = np.arange(0, len(target))
        errors.append(cov_inv.solve(meanval - target)[P])
    
    errors = np.array(errors)
    errors = np.squeeze(errors, axis=0)
    
    return errors
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 05:32:45 2013

@author: tisimst
"""
import random
from copy import copy
import numpy as np
import scipy.stats as ss

def k_fold_cross_validation(X, K, randomise=False):
    """
    Generates K (training, validation) pairs from the items in X.

    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.
    ## {{{ http://code.activestate.com/recipes/521906/ (r3)
    """
    if randomise:
        X = list(X)
        random.shuffle(X)
    for k in range(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation


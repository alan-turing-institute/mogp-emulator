.. _DimensionReduction:

*********************************
The ``DimensionReduction`` module
*********************************

.. automodule:: mogp_emulator.DimensionReduction


---------------------------
Dimension Reduction Classes
---------------------------

.. autoclass:: mogp_emulator.gKDR

    .. automethod:: __init__
    .. automethod:: __call__
    .. automethod:: tune_parameters

---------
Utilities
---------
		    
.. automethod:: mogp_emulator.DimensionReduction.gram_matrix

.. automethod:: mogp_emulator.DimensionReduction.gram_matrix_sqexp

.. automethod:: mogp_emulator.DimensionReduction.median_dist
       
.. rubric:: References
.. [LG17] Liu, Xiaoyu, and Serge Guillas. "Dimension reduction for Gaussian process emulation: An application to the influence of bathymetry on tsunami heights." SIAM/ASA Journal on Uncertainty Quantification 5.1 (2017): 787-812. https://epubs.siam.org/doi/10.1137/16M1090648
.. [Fukumizu1] https://www.ism.ac.jp/~fukumizu/software.html
.. [FL13] Fukumizu, Kenji and Chenlei Leng. "Gradient-based kernel dimension reduction for regression." Journal of the American Statistical Association 109, no. 505 (2014): 359-370

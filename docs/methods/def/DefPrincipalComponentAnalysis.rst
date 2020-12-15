.. _DefPrincipalComponentAnalysis:

Definition of Term: Principal Component Analysis
================================================

Principal Component Analysis (PCA) is a technique used in multivariate
data analysis for dimension reduction. Given a sample of vector random
variables, the aim is to find a set of orthogonal coordinate axes, known
as principal components, with a small number of principal components
describing most of the variation in the data. Projecting the data onto
the first principal component maximises the variance of the projected
points, compared to all other linear projections. Projecting the data
onto the second principal component maximises the variance of the
projected points, compared to all other linear projections that are
orthogonal to the first principal component. Subsequent principal
components are defined likewise.

The principal components are obtained using an eigendecomposition of the
variance matrix of the data. The eigenvectors give the principal
components, and the eigenvalues divided by their sum give the proportion
of variation in the data explained by each associated eigenvector (hence
the eigenvector with the largest eigenvalue is the first principal
component and so on).

If a :ref:`simulator<DefSimulator>` output is multivariate, for
example a time series, PCA can be used to reduce the dimension of the
output before constructing an :ref:`emulator<DefEmulator>`.

# Short demo of how to fit and use the GP class to predict unseen values based on a
# mean function and prior distributions.

# Before loading reticulate, you will need to configure your Python Path to
# use the correct Python version where mogp_emulator is installed.
# mogp_emulator requires Python 3, but some OSs still have Python 2 as the
# default, so you may not get the right one unless you explicitly configure
# it in reticulate. I use the Python that I installed on my Mac with homebrew,
# though on Linux the Python installed via a package manager may have a
# different path.

# The environment variable is RETICULATE_PYTHON, and I set it to
# "/usr/local/bin/python" as this is the Python where mogp_emulator is installed.
# This is set automatically in my .Renviron startup file in my home directory,
# but you may want to configure it some other way. No matter how you decide
# to configure it, you have to set it prior to loading the reticulate library.

library(reticulate)

mogp_emulator <- import("mogp_emulator")
mogp_priors <- import("mogp_emulator.Priors")

# create some data

n_train <- 10
x_scale <- 2.
x1 <- runif(n_train)*x_scale
x2 <- runif(n_train)*x_scale
y <- exp(-x1**2 - x2**2)
x <- data.frame(x1, x2, y)

# GaussianProcess requires data as a matrix, but often you may want to do some
# regression using a data frame in R. To do this, we can split this data frame
# into inputs, targets, and a dictionary mapping column names to integer indices
# using the function below

extract_targets <- function(df, target_cols = list("y")) {
  "separate a data frame into inputs, targets, and inputdict for use with GP class"
  
  for (t in target_cols) {
    stopifnot(t %in% names(x))
  }
  
  n_targets <- length(target_cols)
  
  inputs <- matrix(NA, ncol=ncol(x) - n_targets, nrow=nrow(x))
  targets <- matrix(NA, ncol=n_targets, nrow=nrow(x))
  inputdict <- dict()
  
  input_count <- 1
  target_count <- 1

  for (n in names(x)) {
    if (n %in% target_cols) {
      targets[,target_count] <- as.matrix(x[n])
    } else {
      inputs[,input_count] <- as.matrix(x[n])
      inputdict[n] <- as.integer(input_count - 1)
      input_count <- input_count + 1
    }
  }
  
  if (n_targets == 1) {
    targets <- c(targets)
  }
  
  return(list(inputs, targets, inputdict))
}

target_list <- extract_targets(x)
inputs <- target_list[[1]]
targets <- target_list[[2]]
inputdict <- target_list[[3]]

# Create the mean function formula as a string (or you could extract from the
# formula found via regression). If you want correct expansion of your formula
# in the Python code, you will need to install the patsy package (it is pip
# installable) as it is used internally in mogp_emulator to parse formulas.

# Additionally, you will need to convert the column names from the data frame
# to integer indices in the inputs matrix. This is done with a dict object as
# illustrated below.

mean_func <- "y ~ x1 + x2 + I(x1*x2)"

# Priors are specified by giving a list of prior objects (or NULL if you
# wish to use weak prior information). Each distribution has some parameters
# to set -- NormalPrior is (mean, std), Gamma is (shape, scale), and
# InvGammaPrior is (shape, scale). See the documentation or code for the exact
# functional format of the PDF.

# If you don't know how many parameters you need to specify, it depends on
# the mean function and the number of input dimensions. Mean functions
# have a fixed number of parameters (though in some cases this can depend
# on the dimension of the inputs as well), and then covariance functions have
# one correlation length per input dimension plus a covariance scale and
# a nugget parameter.

# If in doubt, you can create the GP instance with no priors, use gp$n_params
# to get the number, and then set the priors manually using gp$priors <- priors

# In this case, we have 4 mean function parameters (normal distribution on a
# linear scale), 2 correlations lengths (normal distribution on a log scale,
# so lognormal), a sigma^2 covariance parameter (inverse gamma) and a nugget
# (Gamma). If you choose an adaptive or fixed nugget, the nugget prior is ignored.

priors <- list(mogp_priors$NormalPrior(0., 1.),
               mogp_priors$NormalPrior(0., 1.),
               mogp_priors$NormalPrior(0., 1.),
               mogp_priors$NormalPrior(0., 1.),
               mogp_priors$NormalPrior(0., 1.),
               mogp_priors$NormalPrior(0., 1.),
               mogp_priors$InvGammaPrior(2., 1.),
               mogp_priors$GammaPrior(1., 0.2))

# Finally, create the GP instance. If we had multiple outputs, we would
# create a MultiOutputGP class in a similar way, but would have the option
# of giving a single mean and list of priors (assumes it is the same for
# each emulator), or a list of mean functions and a list of lists of
# prior distributions. nugget can also be set with a single value or a list.

gp <- mogp_emulator$GaussianProcess(inputs, targets,
                                    mean=mean_func,
                                    priors=priors,
                                    nugget="fit",
                                    inputdict=inputdict)

# gp is fit using the fit_GP_MAP function. It accepts a GaussianProcess or
# MultiOutputGP object and returns the same type of object with the
# hyperparameters fit via MAP estimation, with some options for how to perform
# the minimization routine. You can also pass the arguments to create a GP/MOGP
# to this function and it will return the object with estimated hyperparameters

gp <- mogp_emulator$fit_GP_MAP(gp)

print(gp$current_logpost)
print(gp$theta)

# now create some test data to make predictions and compare with known values

n_test <- 10000

x1_test <- runif(n_test)*x_scale
x2_test <- runif(n_test)*x_scale

x_test <- cbind(x1_test, x2_test)
y_actual <- exp(-x1_test**2 - x2_test**2)

y_predict <- gp$predict(x_test)

# y_predict is an object holding the mean, variance and derivatives (if computed)
# access the values via y_predict$mean, y_predict$unc, and y_predict$deriv

print(sum((y_actual - y_predict$mean)**2)/n_test)

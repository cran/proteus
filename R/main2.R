#' proteus_random_search
#'
#' @description proteus_random_search is a function for fine-tuning using random search on the hyper-parameter space of proteus (predefined or custom).
#'
#' @param n_samp Positive integer. Number of models to be randomly generated sampling the hyper-parameter space.
#' @param data A data frame with time features on columns and possibly a date column (not mandatory).
#' @param target Vector of strings. Names of the time features to be jointly analyzed.
#' @param future Positive integer. The future dimension with number of time-steps to be predicted.
#' @param past Positive integer. Length of past sequences. Default: NULL (search range future:2*future).
#' @param ci Positive numeric. Confidence interval. Default: 0.8.
#' @param smoother Logical. Perform optimal smoothing using standard loess for each time feature. Default: FALSE.
#' @param t_embed Positive integer. Number of embedding for the temporal dimension. Minimum value is equal to 2. Default: NULL (search range 2:30).
#' @param activ String. Activation function to be used by the forward network. Implemented functions are: "linear", "mish", "swish", "leaky_relu", "celu", "elu", "gelu", "selu", "bent", "softmax", "softmin", "softsign", "softplus", "sigmoid", "tanh". Default: NULL (full-option search).
#' @param nodes Positive integer. Nodes for the forward neural net. Default: NULL (search range 2:1024).
#' @param distr String. Distribution to be used by variational model. Implemented distributions are: "normal", "genbeta", "gev", "gpd", "genray", "cauchy", "exp", "logis", "chisq", "gumbel", "laplace", "lognorm", "skewed". Default: NULL (full-option search).
#' @param optim String. Optimization method. Implemented methods are: "adadelta", "adagrad", "rmsprop", "rprop", "sgd", "asgd", "adam". Default: NULL (full-option search).
#' @param loss_metric String. Loss function for the variational model. Three options: "elbo", "crps", "score". Default: "crps".
#' @param epochs Positive integer. Default: 30.
#' @param lr Positive numeric. Learning rate. Default: NULL (search range 0.001:0.1).
#' @param patience Positive integer. Waiting time (in epochs) before evaluating the overfit performance. Default: epochs.
#' @param latent_sample Positive integer. Number of samples to draw from the latent variables. Default: 100.
#' @param verbose Logical. Default: TRUE
#' @param stride Positive integer. Number of shifting positions for sequence generation. Default: NULL (search range 1:3).
#' @param dates String. Label of feature where dates are located. Default: NULL (progressive numbering).
#' @param rolling_blocks Logical. Option for incremental or rolling window. Default: FALSE.
#' @param n_blocks Positive integer. Number of distinct blocks for back-testing. Default: 4.
#' @param block_minset Positive integer. Minimum number of sequence to create a block. Default: 3.
#' @param error_scale String. Scale for the scaled error metrics (for continuous variables). Two options: "naive" (average of naive one-step absolute error for the historical series) or "deviation" (standard error of the historical series). Default: "naive".
#' @param error_benchmark String. Benchmark for the relative error metrics (for continuous variables). Two options: "naive" (sequential extension of last value) or "average" (mean value of true sequence). Default: "naive".
#' @param batch_size Positive integer. Default: 30.
#' @param min_default Positive numeric. Minimum differentiation iteration. Default: 1.
#' @param seed Random seed. Default: 42.
#' @param future_plan how to resolve the future parallelization. Options are: "future::sequential", "future::multisession", "future::multicore". For more information, take a look at future specific documentation. Default: "future::multisession".
#' @param omit Logical. Flag to TRUE to remove missing values, otherwise all gaps, both in dates and values, will be filled with kalman filter. Default: FALSE.
#' @param keep Logical. Flag to TRUE to keep all the explored models. Default: FALSE.
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#'@return This function returns a list including:
#' \itemize{
#'\item random_search: summary of the sampled hyper-parameters and average error metrics.
#'\item best: best model according to overall ranking on all average error metrics (for negative metrics, absolute value is considered).
#'\item all_models: list with all generated models (if keep flagged to TRUE).
#'\item time_log: computation time.
#' }
#'
#' @export
#'
#' @importFrom fANCOVA loess.as
#' @importFrom imputeTS na_kalman
#' @import purrr
#' @import dplyr
#' @import abind
#' @import torch
#' @import ggplot2
#' @import tictoc
#' @importFrom readr parse_number
#' @import stringr
#' @importFrom lubridate seconds_to_period as_date is.Date
#' @importFrom scales number
#' @importFrom narray split
#' @importFrom utils head tail combn
#' @import torch
#' @importFrom stats ecdf lm median na.omit quantile density rcauchy rchisq rexp rlnorm rlogis rnorm dcauchy dchisq dexp dlnorm dlogis dnorm pcauchy pchisq pexp plnorm plogis pnorm runif sd var
#' @importFrom VGAM rgev pgev dgev rgpd pgpd dgpd rlaplace plaplace dlaplace rgenray pgenray dgenray rgumbel dgumbel pgumbel
#' @importFrom actuar rgenbeta pgenbeta dgenbeta
#' @importFrom modeest mlv1 mfv1
#' @importFrom moments skewness kurtosis
#' @importFrom greybox ME MAE MSE RMSSE MRE MPE MAPE rMAE rRMSE rAME MASE sMSE sCE
#' @importFrom ggthemes theme_clean
#' @import furrr
#' @import future
#'
#' @references https://rpubs.com/giancarlo_vercellino/proteus
#'
proteus_random_search <- function(n_samp, data, target, future, past = NULL, ci = 0.8, smoother = FALSE,
                                  t_embed = NULL, activ = NULL, nodes = NULL, distr = NULL, optim = NULL,
                                  loss_metric = "crps", epochs = 30, lr = NULL, patience = 10, latent_sample = 100, verbose = TRUE,
                                  stride = NULL, dates = NULL, rolling_blocks = FALSE, n_blocks = 4, block_minset = 10,
                                  error_scale = "naive", error_benchmark = "naive", batch_size = 30,
                                  min_default = 1, seed = 42, future_plan = "future::multisession",
                                  omit = FALSE, keep = FALSE)
{
  tic.clearlog()
  tic("random search")

  set.seed(seed)

  past_param <- sampler(past, n_samp, range = future:(2 * future), integer = T)
  embed_param <- sampler(t_embed, n_samp, range = 2:30, integer = T)
  act_param <- sampler(activ, n_samp, range = c("linear", "mish", "swish", "leaky_relu", "celu", "elu", "gelu", "selu", "bent", "softmax", "softmin", "softsign", "softplus", "sigmoid", "tanh"))
  node_param <- sampler(nodes, n_samp, range = c(2:1024), integer = T)
  distr_param <- sampler(distr, n_samp, range = c("normal", "genbeta", "gev", "gpd", "genray", "cauchy", "exp", "logis", "chisq", "gumbel", "laplace", "lognorm", "skewed"))
  opt_param <- sampler(optim, n_samp, range = c("adam", "adagrad", "adadelta", "sgd", "asgd", "rprop", "rmsprop"))
  lrn_param <- round(sampler(lr, n_samp, range = seq(0.001, 0.1, length.out = 1000), integer = F), 3)
  strd_param <- sampler(stride, n_samp, range = 1:3, integer = T)

  hyper_params <- list(past_param, embed_param, act_param, node_param, distr_param, opt_param, lrn_param, strd_param)
  n_cores <- availableCores() - 1
  high_level <- n_cores%/%2
  low_level <- n_cores%/%high_level
  plan(list(tweak(future_plan, workers = high_level), tweak(future_plan, workers = low_level)))

  models <- future_pmap(hyper_params, ~ proteus(data, target, future, past = ..1, ci, smoother,
                                                t_embed = ..2, activ = ..3, nodes = ..4, distr = ..5, optim = ..6,
                                                loss_metric, epochs, lr = ..7, patience, latent_sample, verbose,
                                                stride = ..8, dates, rolling_blocks, n_blocks, block_minset,
                                                error_scale, error_benchmark, batch_size, omit, min_default,
                                                future_plan, seed), .options = furrr_options(seed = T))

  n_feats <- length(target)

  random_search <- data.frame(model = 1:n_samp)
  random_search$past <- past_param
  random_search$t_embed <- embed_param
  random_search$activ <- act_param
  random_search$nodes <- node_param
  random_search$distr <- distr_param
  random_search$optim <- opt_param
  random_search$lr <- lrn_param
  random_search$stride <- strd_param

  avg_errors <- lapply(models, function(x) x$features_errors)
  avg_errors <- map_depth(avg_errors, 2, ~apply(.x, 2, function(x) mean(x[is.finite(x)])))
  if(n_feats > 1){avg_errors <- map(avg_errors, ~ apply(Reduce(rbind, .x), 2, function(x) mean(x[is.finite(x)])))}
  if(n_feats == 1){avg_errors <- flatten(avg_errors)}
  avg_errors <- Reduce(rbind, avg_errors)
  if(n_samp == 1){avg_errors <- t(as.data.frame(avg_errors))}
  colnames(avg_errors) <- paste0("avg_", colnames(avg_errors))

  random_search <- cbind(random_search, round(avg_errors, 4))
  random_search <- ranker(random_search, 10:21, absolute = c("avg_me", "avg_mpe", "avg_sce"))
  rownames(random_search) <- NULL
  best <- models[[head(random_search$model, 1)]]

  toc(log = TRUE)
  time_log<-seconds_to_period(round(parse_number(unlist(tic.log())), 0))

  outcome <- list(random_search = random_search, best = best, time_log = time_log)
  if(keep){outcome$all_models <- models}

  return(outcome)
}

###
sampler <- function(vect, n_samp, range = NULL, integer = FALSE, multi = NULL, variable = NULL, similar = NULL)
{
  if(is.null(vect))
  {
    if(!is.character(range)){if(integer){set <- min(range):max(range)} else {set <- seq(min(range), max(range), length.out = 1000)}} else {set <- range}
    if(is.null(multi) & is.null(variable) & is.null(similar)){samp <- sample(set, n_samp, replace = TRUE)}
    if(is.numeric(multi) & is.null(variable) & is.null(similar)){samp <- replicate(n_samp, sample(set, multi, replace = TRUE), simplify = FALSE)}
    if(is.numeric(variable) & is.null(multi) & is.null(similar)){samp <- replicate(n_samp, sample(set, sample(variable), replace = TRUE), simplify = FALSE)}
    if(!is.null(similar) & is.null(multi) & is.null(variable)){samp <- map(similar, ~ sample(set, length(.x), replace = TRUE), simplify = FALSE)}
  }

  if(!is.null(vect))
  {
    if(is.null(multi) & is.null(variable) & is.null(similar)){
      if(length(vect)==1){samp <- rep(vect, n_samp)}
      if(length(vect) > 1){samp <- sample(vect, n_samp, replace = TRUE)}
    }

    if(is.numeric(multi) & is.null(variable) & is.null(similar)){
      if(length(vect)==1){samp <- replicate(n_samp, rep(vect, multi), simplify = FALSE)}
      if(length(vect) > 1){samp <- replicate(n_samp, sample(vect, multi, replace = TRUE), simplify = FALSE)}
    }

    if(is.numeric(variable)& is.null(multi) & is.null(similar)){samp <- replicate(n_samp, sample(vect, sample(variable), replace = TRUE), simplify = FALSE)}
    if(!is.null(similar) & is.null(multi) & is.null(variable)){samp <- map(similar, ~ sample(vect, length(.x), replace = TRUE), simplify = FALSE)}
  }

  return(samp)
}

###
ranker <- function(df, focus, inverse = NULL, absolute = NULL, reverse = FALSE)
{
  if(nrow(df) == 1){return(df)}
  rank_set <- df[, focus, drop = FALSE]
  if(!is.null(inverse)){rank_set[, inverse] <- - rank_set[, inverse]}###INVERSION BY COL NAMES
  if(!is.null(absolute)){rank_set[, absolute] <- abs(rank_set[, absolute])}###ABS BY COL NAMES
  index <- apply(scale(rank_set), 1, mean, na.rm = TRUE)
  if(reverse == FALSE){df <- df[order(index),]}
  if(reverse == TRUE){df <- df[order(-index),]}
  return(df)
}



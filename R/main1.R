#' proteus
#'
#' @description Proteus is a Sequence-to-Sequence Variational Model designed for time-feature analysis, leveraging a wide range of distributions for improved accuracy. Unlike traditional methods that rely solely on the normal distribution, Proteus uses various latent models to better capture and predict complex processes. To achieve this, Proteus employs a neural network architecture that estimates the shape, location, and scale parameters of the chosen distribution. This approach transforms past sequence data into future sequence parameters, improving the model's prediction capabilities. Proteus also assesses the accuracy of its predictions by estimating the error of measurement and calculating the confidence interval. By utilizing a range of distributions and advanced modeling techniques, Proteus provides a more accurate and comprehensive approach to time-feature analysis.
#'
#' @param data A data frame with time features on columns and possibly a date column (not mandatory)
#' @param target Vector of strings. Names of the time features to be jointly analyzed
#' @param future Positive integer. The future dimension with number of time-steps to be predicted
#' @param past Positive integer. Length of past sequences
#' @param ci Positive numeric. Confidence interval. Default: 0.8
#' @param smoother Logical. Perform optimal smoothing using standard loess for each time feature. Default: FALSE
#' @param t_embed Positive integer. Number of embedding for the temporal dimension. Minimum value is equal to 2. Default: 30.
#' @param activ String. Activation function to be used by the forward network. Implemented functions are: "linear", "mish", "swish", "leaky_relu", "celu", "elu", "gelu", "selu", "bent", "softmax", "softmin", "softsign", "softplus", "sigmoid", "tanh". Default: "linear".
#' @param nodes Positive integer. Nodes for the forward neural net. Default: 32.
#' @param distr String. Distribution to be used by variational model. Implemented distributions are: "normal", "genbeta", "gev", "gpd", "genray", "cauchy", "exp", "logis", "chisq", "gumbel", "laplace", "lognorm", "skewed". Default: "normal".
#' @param optim String. Optimization method. Implemented methods are: "adadelta", "adagrad", "rmsprop", "rprop", "sgd", "asgd", "adam".
#' @param loss_metric String. Loss function for the variational model. Three options: "elbo", "crps", "score". Default: "crps".
#' @param epochs Positive integer. Default: 30.
#' @param lr Positive numeric. Learning rate. Default: 0.01.
#' @param patience Positive integer. Waiting time (in epochs) before evaluating the overfit performance. Default: epochs.
#' @param latent_sample Positive integer. Number of samples to draw from the latent variables. Default: 100.
#' @param verbose Logical. Default: TRUE
#' @param stride Positive integer. Number of shifting positions for sequence generation. Default: 1.
#' @param dates String. Label of feature where dates are located. Default: NULL (progressive numbering).
#' @param rolling_blocks Logical. Option for incremental or rolling window. Default: FALSE.
#' @param n_blocks Positive integer. Number of distinct blocks for back-testing. Default: 4.
#' @param block_minset Positive integer. Minimum number of sequence to create a block. Default: 3.
#' @param error_scale String. Scale for the scaled error metrics. Two options: "naive" (average of naive one-step absolute error for the historical series) or "deviation" (standard error of the historical series). Default: "naive".
#' @param error_benchmark String. Benchmark for the relative error metrics. Two options: "naive" (sequential extension of last value) or "average" (mean value of true sequence). Default: "naive".
#' @param batch_size Positive integer. Default: 30.
#' @param omit Logical. Flag to TRUE to remove missing values, otherwise all gaps, both in dates and values, will be filled with kalman filter. Default: FALSE.
#' @param min_default Positive numeric. Minimum differentiation iteration. Default: 1.
#' @param future_plan how to resolve the future parallelization. Options are: "future::sequential", "future::multisession", "future::multicore". For more information, take a look at future specific documentation. Default: "future::multisession".
#' @param seed Random seed. Default: 42.
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#'
#'@return This function returns a list including:
#' \itemize{
#'\item model_descr: brief model description (number of tensors and parameters)
#'\item prediction: a table with quantile predictions, mean, std, mode, skewness and kurtosis for each time feature (and other metrics, such as iqr_to_range, above_to_below_range, upside_prob, divergence).
#'\item pred_sampler: empirical function for sampling each prediction point for each time feature
#'\item plot: graph with history and prediction for each time feature
#'\item feature_errors: train and test error for each time feature (me, mae, mse, rmsse, mpe, mape, rmae, rrmse, rame, mase, smse, sce)
#'\item history: average cross-validation loss across blocks
#'\item time_log: computation time.
#' }
#'
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
#' @importFrom sn rsn psn dsn
#'
#' @references https://rpubs.com/giancarlo_vercellino/proteus
#'
proteus <- function(data, target, future, past, ci = 0.8, smoother = FALSE,
                    t_embed = 30, activ = "linear", nodes = 32, distr = "normal", optim = "adam",
                    loss_metric = "crps", epochs = 30, lr = 0.01, patience = 10, latent_sample = 100, verbose = TRUE,
                    stride = 1, dates = NULL, rolling_blocks = FALSE, n_blocks = 4, block_minset = 30,
                    error_scale = "naive", error_benchmark = "naive", batch_size = 30, omit = FALSE,
                    min_default = 1, future_plan = "future::multisession", seed = 42)
{

  tic.clearlog()
  tic("proteus")

  ###PRECHECK
  if(cuda_is_available()){dev <- "cuda"} else {dev <- "cpu"}
  deriv <- map_dbl(data[, target, drop = FALSE], ~ best_deriv(.x, min_default = min_default))

  if(max(deriv) >= future | max(deriv) >= past){stop("deriv cannot be equal or greater than future and/or past")}
  if(future <= 0 | past <= 0){stop("past and future must be strictly positive integers")}
  if(t_embed < 2){t_embed <- 2; if(verbose == TRUE){cat("setting t_embed to the minimum possible value (2)\n")}}
  if(n_blocks < 3){n_blocks <- 3; if(verbose == TRUE){cat("setting n_blocks to the minimum possible value (3)\n")}}

  ####PREPARATION & MISSING IMPUTATION
  set.seed(seed)
  torch_manual_seed(seed)

  data <- gap_fixer(data, dates, verbose, omit)
  if(is.null(dates)){time_unit <- NULL} else {time_unit <- units(diff.Date(data[[dates]]))}
  if(is.null(dates)) {date_vector <- NULL} else {date_vector <- data[[dates]]}

  data <- data[, target, drop = FALSE]
  n_feat <- ncol(data)
  n_length <- nrow(data)

  ###SMOOTHING
  if(smoother==TRUE){data <- as.data.frame(map(data, ~ loess.as(x=1:n_length, y=.x)$fitted)); if(verbose == TRUE){cat("performing optimal smoothing\n")}}

  ###SEGMENTATION
  block_model <- block_sampler(data, seq_len = past + future, n_blocks, block_minset, stride)
  block_set <- block_model$block_set
  block_index <- block_model$block_index
  feature_block <- 1:past
  target_block <- (past - max(deriv) + 1):(past + future)

  #train_history_list <- vector("list", n_blocks-1)
  #test_history_list <- vector("list", n_blocks-1)
  #block_features_errors <- vector("list", n_blocks-1)
  #block_raw_errors <- vector("list", n_blocks-1)

  plan(future_plan)

  cross_validation_results <- future_map(1:(n_blocks-1), function(n)
  {
    if(verbose == TRUE){cat("\nblock", n,"\n")}
    if(rolling_blocks == FALSE){train_reframed <- abind(block_set[1:n], along = 1)}
    if(rolling_blocks == TRUE){train_reframed <- abind(block_set[n], along = 1)}
    if(verbose == TRUE){cat(nrow(train_reframed), "sequence for training\n")}

    x_train <- train_reframed[,feature_block,,drop=FALSE]
    y_train <- train_reframed[,target_block,,drop=FALSE]

    test_reframed <- abind(block_set[n+1], along = 1)
    if(verbose == TRUE){cat(nrow(test_reframed), "sequence for testing\n")}

    x_test <- test_reframed[,feature_block,,drop=FALSE]
    y_test <- test_reframed[,target_block,,drop=FALSE]

    new_data <- tail(data, past)
    new_reframed <- block_reframer(new_data, past, stride)

    ###DERIVATIVE
    x_train_deriv_model <- reframed_multiple_differentiation(x_train, deriv)
    x_train <- x_train_deriv_model$reframed
    y_train_deriv_model <- reframed_multiple_differentiation(y_train, deriv)
    y_train <- y_train_deriv_model$reframed

    x_test_deriv_model <- reframed_multiple_differentiation(x_test, deriv)
    x_test <- x_test_deriv_model$reframed
    y_test_deriv_model <- reframed_multiple_differentiation(y_test, deriv)
    y_test <- y_test_deriv_model$reframed

    ###TRAINING MODEL
    model <- nn_variational_model(target_len = future, seq_len = past - max(deriv), n_feat = n_feat, t_embed = t_embed, activ = activ, nodes = nodes, dev = dev, distr = distr)
    training <- training_function(model, x_train, y_train, x_test, y_test, loss_metric, optim, lr, epochs, patience, verbose, batch_size, distr, dev)
    train_history <- training$train_history
    test_history <- training$test_history
    model <- training$model

    pred_train <- pred_fun(model, x_train, "mean", n_sample = latent_sample, dev=dev)
    pred_test <- pred_fun(model, x_test, "mean", n_sample = latent_sample, dev=dev)

    ###INTEGRATION
    pred_train <- reframed_multiple_integration(pred_train, x_train_deriv_model$dmodels, pred = TRUE)
    pred_test <- reframed_multiple_integration(pred_test, x_test_deriv_model$dmodels, pred = TRUE)

    ###ERRORS
    predict_block <- (past + 1):(past + future)
    train_true <- train_reframed[,predict_block,,drop=FALSE]
    train_errors <- pmap(list(smart_split(train_true, along = 3), smart_split(pred_train, along = 3), data), ~ custom_metrics(..1, ..2, actuals = ..3[block_index == n], error_scale, error_benchmark))

    predict_block <- (past + 1):(past + future)
    test_true <- test_reframed[,predict_block,,drop=FALSE]
    test_errors <- pmap(list(smart_split(test_true, along = 3), smart_split(pred_test, along = 3), data), ~ custom_metrics(..1, ..2, actuals = ..3[block_index == n], error_scale, error_benchmark))

    block_raw_errors <- aperm(apply(test_true - pred_test, c(1, 2, 3), mean, na.rm=TRUE), c(2, 1, 3))

    features_errors <- transpose(list(train_errors, test_errors))
    features_errors <- map(features_errors, ~ Reduce(rbind, .x))
    features_errors <- map(features_errors, ~ {rownames(.x) <- c("train", "test"); return(.x)})
    names(features_errors) <- target

    train_history_list <- training$train_history
    test_history_list <- training$test_history

    block_features_errors <- features_errors

    out <- list(train_history_list = train_history_list, test_history_list = test_history_list,
                block_features_errors = block_features_errors, block_raw_errors = block_raw_errors,
                new_reframed = new_reframed)

    return(out)
  }, .options = furrr_options(seed = T))

  cv_transposed <- transpose(cross_validation_results)

  features_errors <- map(transpose(cv_transposed$block_features_errors), ~ Reduce('+', .x)/(n_blocks-1))
  raw_errors <- aperm(abind(cv_transposed$block_raw_errors, along = 2), c(2, 1, 3))

  ###FINAL MODEL
  if(verbose == TRUE){cat("\nfinal training on all", n_blocks,"\n")}
  train_reframed <- abind(block_set[1:n_blocks], along = 1)
  if(verbose == TRUE){cat(nrow(train_reframed), "sequence for training\n")}

  x_train <- train_reframed[,feature_block,,drop=FALSE]
  y_train <- train_reframed[,target_block,,drop=FALSE]

  x_train_deriv_model <- reframed_multiple_differentiation(x_train, deriv)
  x_train <- x_train_deriv_model$reframed
  y_train_deriv_model <- reframed_multiple_differentiation(y_train, deriv)
  y_train <- y_train_deriv_model$reframed

  new_data_deriv_model <- reframed_multiple_differentiation(cv_transposed$new_reframed[[n_blocks-1]], deriv)
  new_reframed <- new_data_deriv_model$reframed

  model <- nn_variational_model(target_len = future, seq_len = past - max(deriv), n_feat = n_feat, t_embed = t_embed, activ = activ, nodes = nodes, dev = dev, distr = distr)
  training <- training_function(model, x_train, y_train, x_test = NULL, y_test = NULL, loss_metric, optim, lr, epochs, patience, verbose, batch_size, distr, dev)
  train_history <- training$train_history
  model <- training$model

  ###LOSS PLOT
  len <- map_dbl(cv_transposed$train_history_list, ~length(.x))
  extend <- max(len) - len
  cv_train_history <- colMeans(Reduce(rbind, map2(cv_transposed$train_history_list, extend, ~ c(.x, rep(smart_tail(.x, 1), .y)))))

  len <- map_dbl(cv_transposed$test_history_list, ~length(.x))
  extend <- max(len) - len
  cv_test_history <- colMeans(Reduce(rbind, map2(cv_transposed$test_history_list, extend, ~c(.x, rep(smart_tail(.x, 1), .y)))))

  act_epochs <- min(c(length(cv_train_history), length(cv_test_history)))
  x_ref_point <- c(quantile(1:act_epochs, 0.15), quantile(1:act_epochs, 0.75))
  y_ref_point <- c(quantile(cv_test_history, 0.75), quantile(cv_train_history, 0.15))

  train_data <- data.frame(epochs = 1:act_epochs, cv_train_history = cv_train_history[1:act_epochs])

  history <- ggplot(train_data) +
    geom_point(aes(x = epochs, y = cv_train_history), col = "blue", shape = 1, size = 1) +
    geom_smooth(col="darkblue", aes(x = epochs, y = cv_train_history), se=FALSE, method = "loess")

  val_data <- data.frame(epochs = 1:act_epochs, cv_test_history = cv_test_history[1:act_epochs])

  history <- history + geom_point(aes(x = epochs, y = cv_test_history), val_data, col = "orange", shape = 1, size = 1) +
    geom_smooth(aes(x = epochs, y = cv_test_history), val_data, col="darkorange", se=FALSE, method = "loess")

  history <- history + ylab("Loss") + xlab("Epochs") +
    annotate("text", x = x_ref_point[1], y = y_ref_point[1], label = "TESTING ERROR", col = "darkorange", hjust = 0, vjust= 0) + ###SINCE THIS IS THE FINAL SET, WE ARE TESTING
    annotate("text", x = x_ref_point[2], y = y_ref_point[2], label = "TRAINING ERROR", col = "darkblue", hjust = 0, vjust= 0) +
    theme_clean() +
    ylab(paste0("average cv error (", loss_metric, ")"))

  ###NEW PREDICTION
  pred_new <- replicate(dim(raw_errors)[1], pred_fun(model, new_reframed, "sample", n_sample = latent_sample, dev=dev), simplify = FALSE)
  pred_new <- map(pred_new, ~ reframed_multiple_integration(.x, new_data_deriv_model$dmodels, pred = TRUE))###UNLIST NEEDED ONLY HERE

  integrated_pred <- abind(pred_new, along=1) + raw_errors
  prediction <- map2(smart_split(integrated_pred, along = 3), data, ~ qpred(.x, ts = .y, ci, error_scale, error_benchmark))
  prediction <- map(prediction, ~{rownames(.x) <- paste0("t", 1:future); return(.x)})
  names(prediction) <- target

  if(!is.null(date_vector))
  {
    start <- as.Date(tail(date_vector, 1))
    new_dates<- seq.Date(from = start, length.out = future, by = time_unit)
    prediction <- map(prediction, ~{rownames(.x) <- as.character(new_dates); return(.x)})
  }

  plot <- pmap(list(data, prediction, target), ~ plotter(quant_pred = ..2, ci, ts = ..1, dates = date_vector, time_unit, feat_name = ..3))

  pred_sampler <- map2(smart_split(integrated_pred, along = 3), data, ~ apply(doxa_filter(.y, .x), 2, function(x) function(n = 1) sample(x, n, replace = TRUE)))
  names(pred_sampler) <- target
  pred_sampler <- map2(pred_sampler, prediction, ~ {names(.x) <- rownames(.y); return(.x)})

  n_tensors <- length(model$parameters)
  n_parameters <- sum(map_dbl(model$parameters, ~ length(as.vector(as_array(.x)))))
  if(verbose==TRUE){cat("\nvariational model based on", distr, "latent distribution with", n_tensors, "tensors and", n_parameters, "parameters\n")}
  model_descr <- paste0("variational model based on ", distr, " latent distribution with ", n_tensors, " tensors and ", n_parameters, " parameters")

  toc(log = TRUE)
  time_log<-seconds_to_period(round(parse_number(unlist(tic.log())), 0))

  ###OUTCOMES
  outcome <- list(model_descr = model_descr, prediction = prediction, pred_sampler = pred_sampler, plot = plot, features_errors = features_errors, history = history, time_log = time_log)

  return(outcome)

}

###SUPPORT FUNCTIONS
nn_mish <- nn_module(
  "nn_mish",
  initialize = function() {self$softplus <- nn_softplus(beta = 1)},
  forward = function(x) {x * torch_tanh(self$softplus(x))})

nn_bent <- nn_module(
  "nn_bent",
  initialize = function() {},
  forward = function(x) {(torch_sqrt(x^2 + 1) - 1)/2 + x})

nn_swish <- nn_module(
  "nn_swish",
  initialize = function(beta = 1) {self$beta <- nn_buffer(beta)},
  forward = function(x) {x * torch_sigmoid(self$beta * x)})

nn_activ <- nn_module(
  "nn_activ",
  initialize = function(act, dim = 2)
  {
    if(act == "linear"){self$activ <- nn_identity()}
    if(act == "mish"){self$activ <- nn_mish()}
    if(act == "leaky_relu"){self$activ <- nn_leaky_relu()}
    if(act == "celu"){self$activ <- nn_celu()}
    if(act == "elu"){self$activ <- nn_elu()}
    if(act == "gelu"){self$activ <- nn_gelu()}
    if(act == "selu"){self$activ <- nn_selu()}
    if(act == "softplus"){self$activ <- nn_softplus()}
    if(act == "bent"){self$activ <- nn_bent()}
    if(act == "softmax"){self$activ <- nn_softmax(dim)}
    if(act == "softmin"){self$activ <- nn_softmin(dim)}
    if(act == "softsign"){self$activ <- nn_softsign()}
    if(act == "sigmoid"){self$activ <- nn_sigmoid()}
    if(act == "tanh"){self$activ <- nn_tanh()}
    if(act == "swish"){self$activ <- nn_swish()}
  },
  forward = function(x)
  {
  x <- self$activ(x)
  })


######
nn_normalization <- nn_module(
  "nn_normalization",
  initialize = function(seq_len, dev)
  {
    self$shift <- nn_linear(in_features = seq_len, out_features = seq_len, bias = TRUE)
    self$scale <- nn_linear(in_features = seq_len, out_features = seq_len, bias = TRUE)
    self$adapt <- nn_linear(in_features = seq_len, out_features = seq_len, bias = TRUE)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)
    x <- torch_transpose(x, 4, 2)
    x <- (x - self$shift(x))/self$scale(x)
    x <- torch_sigmoid(self$adapt(x))
    x <- torch_transpose(x, 4, 2)
  })

nn_time_transformation <- nn_module(
  "nn_time_transformation",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$norm <- nn_normalization(seq_len, dev)
    self$trend <- nn_linear(seq_len, seq_len, bias = TRUE)
    pnames <- paste0("periodic", 1:(t_embed - 1))
    map(pnames, ~ {self[[.x]] <- nn_linear(seq_len, seq_len, bias = TRUE)})
    self$pnames <- nn_buffer(pnames)
    self$dev <- nn_buffer(dev)

    self$fnet <- nn_linear(t_embed, nodes, bias = TRUE)
    self$focus <- nn_linear(nodes, 1, bias = TRUE)
    self$target <- nn_linear(seq_len, target_len, bias = TRUE)
    self$activ <- nn_activ(act = activ)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)
    x <- torch_transpose(x, 3, 2)
    trend_embedding <- self$trend(x)
    pnames <- self$pnames
    periodic_embedding <- map(pnames, ~ torch_cos(self[[.x]](x)))
    time_embeddings <- prepend(periodic_embedding, trend_embedding)
    time_embeddings <- map(time_embeddings, ~ torch_transpose(.x, 3, 2))
    time_embeddings <- map(time_embeddings, ~ .x$unsqueeze(4))
    time_embeddings <- torch_sigmoid(torch_cat(time_embeddings, dim = 4))
    interim <- self$norm(time_embeddings)
    interim <- self$activ(self$fnet(interim))
    interim <- self$activ(self$focus(interim))$squeeze(4)
    interim <- self$activ(self$target(torch_transpose(interim, 3, 2)))
    result <- torch_transpose(interim, 3, 2)
  })


nn_variational_model <- nn_module(
  "nn_variational_model",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev, distr)
  {
    self$distr <- nn_buffer(distr)
    if(distr == "normal"){self$var_model <- nn_normal_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
    if(distr == "genbeta"){self$var_model <- nn_genbeta_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
    if(distr == "gev"){self$var_model <- nn_gev_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
    if(distr == "gpd"){self$var_model <- nn_gpd_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
    if(distr == "genray"){self$var_model <- nn_genray_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
    if(distr == "cauchy"){self$var_model <- nn_cauchy_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
    if(distr == "exp"){self$var_model <- nn_exp_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
    if(distr == "logis"){self$var_model <- nn_logis_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
    if(distr == "chisq"){self$var_model <- nn_chisq_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
    if(distr == "gumbel"){self$var_model <- nn_gumbel_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
    if(distr == "laplace"){self$var_model <- nn_laplace_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
    if(distr == "lognorm"){self$var_model <- nn_lognorm_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
    if(distr == "skewed"){self$var_model <- nn_skewed_layer(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)}
  },
  forward = function(x)
  {
    distr <- self$distr
    result <- self$var_model(x)
    outcome <- list(latent = result[[1]], params = result[-1], distr = distr)
    return(outcome)
  })

nn_skewed_layer <- nn_module(
  "nn_skewed_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$xi_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$omega_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$alpha_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    xi_param <- self$xi_param(x)
    omega_param <- torch_square(self$omega_param(x))
    alpha_param <- self$alpha_param(x)

    latent <- tensor_apply(rsn, values = NULL, "xi" = as_array(xi_param), "omega" = as_array(omega_param), "alpha" = as_array(alpha_param))
    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, xi_param = xi_param, omega_param = omega_param, alpha_param = alpha_param)
    return(outcome)
  })


nn_normal_layer <- nn_module(
  "nn_normal_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$mean_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$scale_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    mean_param <- self$mean_param(x)
    scale_param <- torch_square(self$scale_param(x))

    latent <- tensor_apply(rnorm, values = NULL, "mean" = as_array(mean_param), "sd" = as_array(scale_param))
    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, mean_param = mean_param, scale_param = scale_param)
    return(outcome)
  })

nn_genbeta_layer <- nn_module(
  "nn_genbeta_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$shape1_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$shape2_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$shape3_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    shape1_param <- torch_square(self$shape1_param(x))
    shape2_param <- torch_square(self$shape2_param(x))
    shape3_param <- torch_square(self$shape3_param(x))

    latent <- tensor_apply(rgenbeta, values = NULL, "shape1" = as_array(shape1_param), "shape2" = as_array(shape2_param), "shape3" = as_array(shape3_param))
    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, shape1_param = shape1_param, shape2_param = shape2_param, shape3_param = shape3_param)
    return(outcome)
  })

nn_gev_layer <- nn_module(
  "nn_gev_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$location_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$scale_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$shape_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    location_param <- self$location_param(x)
    scale_param <- torch_square(self$scale_param(x))
    shape_param <- self$shape_param(x)

    latent <- tensor_apply(rgev, values = NULL, "location" = as_array(location_param), "scale" = as_array(scale_param), "shape" = as_array(shape_param))
    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, location_param = location_param, scale_param = scale_param, shape_param = shape_param)
    return(outcome)
  })

nn_gpd_layer <- nn_module(
  "nn_gpd_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$location_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$scale_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$shape_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    location_param <- self$location_param(x)
    scale_param <- torch_square(self$scale_param(x))
    shape_param <- self$shape_param(x)

    latent <- tensor_apply(rgpd, values = NULL, "location" = as_array(location_param), "scale" = as_array(scale_param), "shape" = as_array(shape_param))
    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, location_param = location_param, scale_param = scale_param, shape_param = shape_param)
    return(outcome)
  })


nn_genray_layer <- nn_module(
  "nn_genray_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$scale_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$shape_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    scale_param <- torch_square(self$scale_param(x))
    shape_param <- torch_square(self$shape_param(x))

    latent <- tensor_apply(rgenray, values = NULL, "scale" = as_array(scale_param), "shape" = as_array(shape_param))
    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, scale_param = scale_param, shape_param = shape_param)
    return(outcome)
  })

nn_cauchy_layer <- nn_module(
  "nn_cauchy_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$location_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$scale_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    location_param <- self$location_param(x)
    scale_param <- torch_square(self$scale_param(x))

    latent <- tensor_apply(rcauchy, values = NULL, "location" = as_array(location_param), "scale" = as_array(scale_param))
    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, location_param = location_param, scale_param = scale_param)
    return(outcome)
  })

nn_exp_layer <- nn_module(
  "nn_exp_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$rate_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    rate_param <- torch_square(self$rate_param(x))

    latent <- tensor_apply(rexp, values = NULL, "rate" = as_array(rate_param))
    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, rate_param = rate_param)
    return(outcome)
  })

nn_logis_layer <- nn_module(
  "nn_logis_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$location_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$scale_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    location_param <- self$location_param(x)
    scale_param <- torch_square(self$scale_param(x))

    latent <- tensor_apply(rlogis, values = NULL, "location" = as_array(location_param), "scale" = as_array(scale_param))
    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, location_param = location_param, scale_param = scale_param)
    return(outcome)
  })


nn_chisq_layer <- nn_module(
  "nn_chisq_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$df_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$ncp_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    df_param <- torch_square(self$df_param(x))
    ncp_param <- torch_square(self$ncp_param(x))

    latent <- tensor_apply(rchisq, values = NULL, "df" = as_array(df_param), "ncp" = as_array(ncp_param))

    if(any(is.infinite(latent))){latent <- abind(map(smart_split(latent, along = 3), ~ {.x[.x == Inf] <- max(.x[is.finite(.x)]); .x[.x== -Inf] <- min(.x[is.finite(.x)]); return(.x)}), along = 3)}

    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, df_param = df_param, ncp_param = ncp_param)
    return(outcome)
  })

nn_gumbel_layer <- nn_module(
  "nn_gumbel_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$location_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$scale_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    location_param <- self$location_param(x)
    scale_param <- torch_square(self$scale_param(x))

    latent <- tensor_apply(rgumbel, values = NULL, "location" = as_array(location_param), "scale" = as_array(scale_param))

    if(any(is.infinite(latent))){latent <- abind(map(smart_split(latent, along = 3), ~ {.x[.x == Inf] <- max(.x[is.finite(.x)]); .x[.x== -Inf] <- min(.x[is.finite(.x)]); return(.x)}), along = 3)}

    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, location_param = location_param, scale_param = scale_param)
    return(outcome)
  })


nn_laplace_layer <- nn_module(
  "nn_laplace_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$location_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$scale_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    location_param <- self$location_param(x)
    scale_param <- torch_square(self$scale_param(x))

    latent <- tensor_apply(rlaplace, values = NULL, "location" = as_array(location_param), "scale" = as_array(scale_param))
    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, location_param = location_param, scale_param = scale_param)
    return(outcome)
  })

nn_lognorm_layer <- nn_module(
  "nn_lognorm_layer",
  initialize = function(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
  {
    self$meanlog_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$sdlog_param <- nn_time_transformation(target_len, seq_len, n_feat, t_embed, activ, nodes, dev)
    self$dev <- nn_buffer(dev)
  },
  forward = function(x)
  {
    dev <- self$dev
    torch_device(dev)

    meanlog_param <- self$meanlog_param(x)
    sdlog_param <- torch_square(self$sdlog_param(x))

    latent <- tensor_apply(rlnorm, values = NULL, "meanlog" = as_array(meanlog_param), "sdlog" = as_array(sdlog_param))
    latent <- torch_tensor(latent)

    outcome <- list(latent = latent, meanlog_param = meanlog_param, sdlog_param = sdlog_param)
    return(outcome)
  })


####
crps_loss <- function(actual, latent, params, distr, dev)
{
  ###CONTINUOUS RANKED PROBABILITY SCORE (CRPS)
  if(distr == "normal"){latent_cdf <- pnorm(as_array(latent$cpu()), mean = as_array(params[[1]]$cpu()), sd = as_array(params[[2]]$cpu()))}
  if(distr == "genbeta"){latent_cdf <- pgenbeta(as_array(latent$cpu()), shape1 = as_array(params[[1]]$cpu()), shape2 = as_array(params[[2]]$cpu()), shape3 = as_array(params[[3]]$cpu()))}
  if(distr == "gev"){latent_cdf <- pgev(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), shape = as_array(params[[3]]$cpu()))}
  if(distr == "gpd"){latent_cdf <- pgpd(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), shape = as_array(params[[3]]$cpu()))}
  if(distr == "genray"){latent_cdf <- pgenray(as_array(latent$cpu()), scale = as_array(params[[1]]$cpu()), shape = as_array(params[[2]]$cpu()))}
  if(distr == "cauchy"){latent_cdf <- pcauchy(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()))}
  if(distr == "exp"){latent_cdf <- pexp(as_array(latent$cpu()), rate = as_array(params[[1]]$cpu()))}
  if(distr == "logis"){latent_cdf <- plogis(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()))}
  if(distr == "chisq"){latent_cdf <- pchisq(as_array(latent$cpu()), df = as_array(params[[1]]$cpu()), ncp = as_array(params[[2]]$cpu()))}
  if(distr == "gumbel"){latent_cdf <- pgumbel(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()))}
  if(distr == "laplace"){latent_cdf <- plaplace(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()))}
  if(distr == "lognorm"){latent_cdf <- plnorm(as_array(latent$cpu()), meanlog = as_array(params[[1]]$cpu()), sdlog = as_array(params[[2]]$cpu()))}
  if(distr == "skewed"){latent_cdf <- psn(as_array(latent$cpu()), xi = as_array(params[[1]]$cpu()), omega = as_array(params[[2]]$cpu()), alpha = as_array(params[[3]]$cpu()))}
  error <- as_array(latent$cpu()) - as_array(actual$cpu())
  heaviside_step <- error >= 0
  loss <- mean((latent_cdf - heaviside_step)^2)###MEAN INSTEAD OF SUM

  torch_device(dev)
  loss <- torch_tensor(loss, dtype = torch_float64(), requires_grad = TRUE)

  return(loss)
}

###
score_loss <- function(actual, latent, params, distr, dev)
{
  if(distr == "normal"){scores <- pnorm(as_array(actual$cpu()), mean = as_array(params[[1]]$cpu()), sd = as_array(params[[2]]$cpu())) - pnorm(as_array(latent$cpu()), mean = as_array(params[[1]]$cpu()), sd = as_array(params[[2]]$cpu()))}
  if(distr == "genbeta"){scores <- pgenbeta(as_array(actual$cpu()), shape1 = as_array(params[[1]]$cpu()), shape2 = as_array(params[[2]]$cpu()), shape3 = as_array(params[[3]]$cpu())) - pgenbeta(as_array(latent$cpu()), shape1 = as_array(params[[1]]$cpu()), shape2 = as_array(params[[2]]$cpu()), shape3 = as_array(params[[3]]$cpu()))}
  if(distr == "gev"){scores <- pgev(as_array(actual$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), shape = as_array(params[[3]]$cpu())) - pgev(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), shape = as_array(params[[3]]$cpu()))}
  if(distr == "gpd"){scores <- pgpd(as_array(actual$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), shape = as_array(params[[3]]$cpu())) - pgpd(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), shape = as_array(params[[3]]$cpu()))}
  if(distr == "genray"){scores <- pgenray(as_array(actual$cpu()), scale = as_array(params[[1]]$cpu()), shape = as_array(params[[2]]$cpu())) - pgenray(as_array(latent$cpu()), scale = as_array(params[[1]]$cpu()), shape = as_array(params[[2]]$cpu()))}
  if(distr == "cauchy"){scores <- pcauchy(as_array(actual$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu())) - pcauchy(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()))}
  if(distr == "exp"){scores <- pexp(as_array(actual$cpu()), rate = as_array(params[[1]]$cpu())) - pexp(as_array(latent$cpu()), rate = as_array(params[[1]]$cpu()))}
  if(distr == "logis"){scores <- plogis(as_array(actual$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu())) - plogis(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()))}
  if(distr == "chisq"){scores <- pchisq(as_array(actual$cpu()), df = as_array(params[[1]]$cpu()), ncp = as_array(params[[2]]$cpu())) - pchisq(as_array(latent$cpu()), df = as_array(params[[1]]$cpu()), ncp = as_array(params[[2]]$cpu()))}
  if(distr == "gumbel"){scores <- pgumbel(as_array(actual$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu())) - pgumbel(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()))}
  if(distr == "laplace"){scores <- plaplace(as_array(actual$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu())) - plaplace(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()))}
  if(distr == "lognorm"){scores <- plnorm(as_array(actual$cpu()), meanlog = as_array(params[[1]]$cpu()), sdlog = as_array(params[[2]]$cpu())) - plnorm(as_array(latent$cpu()), meanlog = as_array(params[[1]]$cpu()), sdlog = as_array(params[[2]]$cpu()))}
  if(distr == "skewed"){scores <- psn(as_array(actual$cpu()), xi = as_array(params[[1]]$cpu()), omega = as_array(params[[2]]$cpu()), alpha = as_array(params[[3]]$cpu())) - psn(as_array(latent$cpu()), xi = as_array(params[[1]]$cpu()), omega = as_array(params[[2]]$cpu()), alpha = as_array(params[[3]]$cpu()))}

  loss <- mean(abs(scores[is.finite(scores)]), na.rm = TRUE)
  torch_device(dev)
  loss <- torch_tensor(loss, dtype = torch_float64(), requires_grad = TRUE)

  return(loss)
}


elbo_loss <- function(actual, latent, params, distr, dev)
{
  ###EVIDENCE LOWER BOUND (ELBO)
  recon <- nnf_l1_loss(input = latent, target = actual, reduction = "none")

  if(distr == "normal")
  {
    latent_pdf <- dnorm(as_array(latent$cpu()), mean = as_array(params[[1]]$cpu()), sd = as_array(params[[2]]$cpu()), log = FALSE)
    log_latent_pdf <- dnorm(as_array(latent$cpu()), mean = as_array(params[[1]]$cpu()), sd = as_array(params[[2]]$cpu()), log = TRUE)
    log_actual_pdf <- dnorm(as_array(actual$cpu()), mean = as_array(params[[1]]$cpu()), sd = as_array(params[[2]]$cpu()), log = TRUE)
  }

  if(distr == "genbeta")
  {
    latent_pdf <- dgenbeta(as_array(latent$cpu()), shape1 = as_array(params[[1]]$cpu()), shape2 = as_array(params[[2]]$cpu()), shape3 = as_array(params[[3]]$cpu()), log = FALSE)
    log_latent_pdf <- dgenbeta(as_array(latent$cpu()), shape1 = as_array(params[[1]]$cpu()), shape2 = as_array(params[[2]]$cpu()), shape3 = as_array(params[[3]]$cpu()), log = TRUE)
    log_actual_pdf <- dgenbeta(as_array(actual$cpu()), shape1 = as_array(params[[1]]$cpu()), shape2 = as_array(params[[2]]$cpu()), shape3 = as_array(params[[3]]$cpu()), log = TRUE)
  }

  if(distr == "gev")
  {
    latent_pdf <- dgev(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), shape = as_array(params[[3]]$cpu()), log = FALSE)
    log_latent_pdf <- dgev(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), shape = as_array(params[[3]]$cpu()), log = TRUE)
    log_actual_pdf <- dgev(as_array(actual$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), shape = as_array(params[[3]]$cpu()), log = TRUE)
  }

  if(distr == "gpd")
  {
    latent_pdf <- dgpd(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), shape = as_array(params[[3]]$cpu()), log = FALSE)
    log_latent_pdf <- dgpd(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), shape = as_array(params[[3]]$cpu()), log = TRUE)
    log_actual_pdf <- dgpd(as_array(actual$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), shape = as_array(params[[3]]$cpu()), log = TRUE)
  }

  if(distr == "genray")
  {
    latent_pdf <- dgenray(as_array(latent$cpu()), scale = as_array(params[[1]]$cpu()), shape = as_array(params[[2]]$cpu()), log = FALSE)
    log_latent_pdf <- dgenray(as_array(latent$cpu()), scale = as_array(params[[1]]$cpu()), shape = as_array(params[[2]]$cpu()), log = TRUE)
    log_actual_pdf <- dgenray(as_array(actual$cpu()), scale = as_array(params[[1]]$cpu()), shape = as_array(params[[2]]$cpu()), log = TRUE)
  }

  if(distr == "cauchy")
  {
    latent_pdf <- dcauchy(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), log = FALSE)
    log_latent_pdf <- dcauchy(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), log = TRUE)
    log_actual_pdf <- dcauchy(as_array(actual$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), log = TRUE)
  }

  if(distr == "exp")
  {
    latent_pdf <- dexp(as_array(latent$cpu()), rate = as_array(params[[1]]$cpu()), log = FALSE)
    log_latent_pdf <- dexp(as_array(latent$cpu()), rate = as_array(params[[1]]$cpu()), log = TRUE)
    log_actual_pdf <- dexp(as_array(actual$cpu()), rate = as_array(params[[1]]$cpu()), log = TRUE)
  }

  if(distr == "logis")
  {
    latent_pdf <- dlogis(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), log = FALSE)
    log_latent_pdf <- dlogis(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), log = TRUE)
    log_actual_pdf <- dlogis(as_array(actual$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), log = TRUE)
  }

  if(distr == "chisq")
  {
    latent_pdf <- dchisq(as_array(latent$cpu()), df = as_array(params[[1]]$cpu()), ncp = as_array(params[[2]]$cpu()), log = FALSE)
    log_latent_pdf <- dchisq(as_array(latent$cpu()), df = as_array(params[[1]]$cpu()), ncp = as_array(params[[2]]$cpu()), log = TRUE)
    log_actual_pdf <- dchisq(as_array(actual$cpu()), df = as_array(params[[1]]$cpu()), ncp = as_array(params[[2]]$cpu()), log = TRUE)
  }

  if(distr == "gumbel")
  {
    latent_pdf <- dgumbel(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), log = FALSE)
    log_latent_pdf <- dgumbel(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), log = TRUE)
    log_actual_pdf <- dgumbel(as_array(actual$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), log = TRUE)
  }


  if(distr == "laplace")
  {
    latent_pdf <- dlaplace(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), log = FALSE)
    log_latent_pdf <- dlaplace(as_array(latent$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), log = TRUE)
    log_actual_pdf <- dlaplace(as_array(actual$cpu()), location = as_array(params[[1]]$cpu()), scale = as_array(params[[2]]$cpu()), log = TRUE)
  }

  if(distr == "lognorm")
  {
    latent_pdf <- dlnorm(as_array(latent$cpu()), meanlog = as_array(params[[1]]$cpu()), sdlog = as_array(params[[2]]$cpu()), log = FALSE)
    log_latent_pdf <- dlnorm(as_array(latent$cpu()), meanlog = as_array(params[[1]]$cpu()), sdlog = as_array(params[[2]]$cpu()), log = TRUE)
    log_actual_pdf <- dlnorm(as_array(actual$cpu()), meanlog = as_array(params[[1]]$cpu()), sdlog = as_array(params[[2]]$cpu()), log = TRUE)
  }

  if(distr == "skewed")
  {
    latent_pdf <- dsn(as_array(latent$cpu()), xi = as_array(params[[1]]$cpu()), omega = as_array(params[[2]]$cpu()), alpha = as_array(params[[3]]$cpu()), log = FALSE)
    log_latent_pdf <- dsn(as_array(latent$cpu()), xi = as_array(params[[1]]$cpu()), omega = as_array(params[[2]]$cpu()), alpha = as_array(params[[3]]$cpu()), log = TRUE)
    log_actual_pdf <- dsn(as_array(actual$cpu()), xi = as_array(params[[1]]$cpu()), omega = as_array(params[[2]]$cpu()), alpha = as_array(params[[3]]$cpu()), log = TRUE)
  }

  elbo <- abs(log_actual_pdf * latent_pdf - log_latent_pdf * latent_pdf)
  elbo <- mean(elbo[is.finite(elbo)])
  recon <- as_array(recon$cpu())
  recon <- mean(recon[is.finite(recon)])

  torch_device(dev)
  loss <- torch_tensor(recon + elbo, dtype = torch_float64(), requires_grad = TRUE)

  return(loss)
}

###
tensor_apply <- function(fun, values = NULL, ...)
{

  dims <- dim(list(...)[[1]])
  custom <- partial(fun, ...)

  functions <- list(function(x) array(x, dims), custom)

  if(is.null(values)){values <- prod(dims)}
  result <- compose(!!!functions)(values)

  return(result)
}

###PREDICTION
pred_fun <- function(model, new_data, type = "sample", quant = 0.5, seed = as.numeric(Sys.time()), n_sample, dev)
{
  torch_device(dev)
  if(!("torch_tensor" %in% class(new_data))){new_data <- torch_tensor(as.array(new_data))}
  if(type=="sample"){pred <- as_array(model(new_data)$latent$cpu())}
  if(type=="set"){pred <- abind(replicate(n = n_sample, as_array(model(new_data)$latent$cpu()), simplify = FALSE), along=-1)}###DRAWING SAMPLES FROM LATENT

  if(type == "quant" | type == "mean" | type == "mode")
  {
    pred <- abind(replicate(n = n_sample, as_array(model(new_data)$latent$cpu()), simplify = FALSE), along=-1)
    if(type == "quant"){pred <- apply(pred, c(2, 3, 4), quantile, probs = quant, na.rm = TRUE)}
    if(type == "mean"){pred <- apply(pred, c(2, 3, 4), mean, na.rm = TRUE)}
    if(type == "mode"){pred <- apply(pred, c(2, 3, 4), function(x) mlv1(x, method = "parzen"))}
  }

  return(pred)
}

####
smart_split <- function(array, along)
{
  array_split <- split(array, along = along)
  dim_checkout <- dim(array)
  dim_preserve <- dim_checkout[- along]
  array_split <- map(array_split, ~ {dim(.x) <- dim_preserve; return(.x)})
  return(array_split)
}


####
training_function <- function(model, x_train, y_train, x_test = NULL, y_test = NULL, loss_metric, optim, lr, epochs, patience, verbose, batch_size, distr, dev)
{
  if(optim == "adadelta"){optimizer <- optim_adadelta(model$parameters, lr = lr, rho = 0.9, eps = 1e-06, weight_decay = 0)}
  if(optim == "adagrad"){optimizer <- optim_adagrad(model$parameters, lr = lr, lr_decay = 0, weight_decay = 0, initial_accumulator_value = 0, eps = 1e-10)}
  if(optim == "rmsprop"){optimizer <- optim_rmsprop(model$parameters, lr = lr, alpha = 0.99, eps = 1e-08, weight_decay = 0, momentum = 0, centered = FALSE)}
  if(optim == "rprop"){optimizer <- optim_rprop(model$parameters, lr = lr, etas = c(0.5, 1.2), step_sizes = c(1e-06, 50))}
  if(optim == "sgd"){optimizer <- optim_sgd(model$parameters, lr = lr, momentum = 0, dampening = 0, weight_decay = 0, nesterov = FALSE)}
  if(optim == "asgd"){optimizer <- optim_asgd(model$parameters, lr = lr, lambda = 1e-04, alpha = 0.75, t0 = 1e+06, weight_decay = 0)}
  if(optim == "adam"){optimizer <- optim_adam(model$parameters, lr = lr, betas = c(0.9, 0.999), eps = 1e-08, weight_decay = 0, amsgrad = FALSE)}

  if(is.null(x_test) && is.null(y_test)) {testing <- FALSE} else {testing <- TRUE}

  train_history <- vector(mode="numeric", length = epochs)
  test_history <- NULL

  if(testing == TRUE)
  {
    test_history <- vector(mode="numeric", length = epochs)
    dynamic_overfit <- vector(mode="numeric", length = epochs)
  }

  torch_device(dev)
  x_train <- torch_tensor(x_train, dtype = torch_float32())
  y_train <- torch_tensor(y_train, dtype = torch_float32())
  n_train <- nrow(x_train)

  if(testing == TRUE)
  {
    x_test <- torch_tensor(x_test, dtype = torch_float32())
    y_test <- torch_tensor(y_test, dtype = torch_float32())
    n_test <- nrow(x_test)
  }

  if(testing == FALSE){if(batch_size > n_train){batch_size <- n_train; if(verbose == TRUE) {cat("setting max batch size to", batch_size,"\n")}}}
  if(testing == TRUE){if(batch_size > n_train | batch_size > n_test){batch_size <- min(c(n_train, n_test)); if(verbose == TRUE) {cat("setting max batch size to", batch_size,"\n")}}}

  train_batches <- ceiling(n_train/batch_size)
  train_batch_index <- c(rep(1:ifelse(n_train%%batch_size==0, train_batches, train_batches-1), each = batch_size), rep(train_batches, each = n_train%%batch_size))

  parameters <- list()

  for(t in 1:epochs)
  {
    train_batch_history <- vector(mode="numeric", length = train_batches)

    for(b in 1:train_batches)
    {
      index <- b == train_batch_index
      train_results <- model(x_train[index,,])
      if(loss_metric == "elbo"){train_loss <- elbo_loss(actual = y_train[index,,], train_results$latent, train_results$params, train_results$distr, dev)}
      if(loss_metric == "crps"){train_loss <- crps_loss(actual = y_train[index,,], train_results$latent, train_results$params, train_results$distr, dev)}
      if(loss_metric == "score"){train_loss <- score_loss(actual = y_train[index,,], train_results$latent, train_results$params, train_results$distr, dev)}
      train_batch_history[b] <- train_loss$item()

      train_loss$backward()
      optimizer$step()
      optimizer$zero_grad()
    }

    train_history[t] <- mean(train_batch_history)

    if(testing == TRUE)
    {
      test_batches <- ceiling(n_test/batch_size)
      test_batch_history <- vector(mode="numeric", length = test_batches)
      test_batch_index <- c(rep(1:ifelse(n_test%%batch_size==0, test_batches, test_batches-1), each = batch_size), rep(test_batches, each = n_test%%batch_size))

      for(b in 1:test_batches)
      {
        index <- b == test_batch_index
        test_results <- model(x_test[index,,])
        if(loss_metric == "elbo"){test_loss <- elbo_loss(actual = y_test[index,,], test_results$latent, test_results$params, test_results$distr, dev)}
        if(loss_metric == "crps"){test_loss <- crps_loss(actual = y_test[index,,], test_results$latent, test_results$params, test_results$distr, dev)}
        if(loss_metric == "score"){test_loss <- score_loss(actual = y_test[index,,], test_results$latent, test_results$params, test_results$distr, dev)}
        test_batch_history[b] <- test_loss$item()
      }

      test_history[t] <- mean(test_batch_history)
    }

    if(verbose == TRUE && testing == TRUE){if (t %% floor(epochs/10) == 0 | epochs < 10) {cat("epoch: ", t, "   Train loss: ", train_history[t], "   Test loss: ", test_history[t], "\n")}}
    if(verbose == TRUE && testing == FALSE){if (t %% floor(epochs/10) == 0 | epochs < 10) {cat("epoch: ", t, "   Train loss: ", train_history[t], "\n")}}

    if(testing == TRUE)
    {
      dynamic_overfit[t] <- abs(test_history[t] - train_history[t])/abs(test_history[1] - train_history[1])
      dyn_ovft_horizon <- c(0, diff(dynamic_overfit[1:t]))
      test_hist_horizon <- c(0, diff(test_history[1:t]))

      if(t >= patience){
        lm_mod1 <- lm(h ~ t, data.frame(t=1:t, h=dyn_ovft_horizon))
        lm_mod2 <- lm(h ~ t, data.frame(t=1:t, h=test_hist_horizon))

        rolling_window <- max(c(patience - t + 1, 1))
        avg_dyn_ovft_deriv <- mean(tail(dyn_ovft_horizon, rolling_window), na.rm = TRUE)
        avg_val_hist_deriv <- mean(tail(test_hist_horizon, rolling_window), na.rm = TRUE)
      }
      if(t >= patience && avg_dyn_ovft_deriv > 0 && lm_mod1$coefficients[2] > 0 && avg_val_hist_deriv > 0 && lm_mod2$coefficients[2] > 0){if(verbose == TRUE){cat("early stop at epoch: ", t, "   Train loss: ", train_loss$item(), "   Test loss: ", test_loss$item(), "\n")}; break}
    }
  }

  outcome <- list(model = model, train_history = train_history[1:t], test_history = test_history[1:t])

  return(outcome)
}


###
reframed_differentiation <- function(reframed, diff)
{
  ddim <- dim(reframed)[2]

  if(diff == 0){return(list(reframed = reframed, head_list = NULL, tail_list = NULL))}

  if(diff > 0)
  {
    if(diff >= ddim){stop("diff is greater/ equal to dim 2")}
    head_list <- list()
    tail_list <- list()

    for(d in 1:diff)
    {
      head_list <- append(head_list, list(reframed[, 1,, drop = FALSE]))
      tail_list <- append(tail_list,  list(reframed[, dim(reframed)[2],, drop = FALSE]))
      reframed <- reframed[,2:(ddim - d + 1),, drop = FALSE] - reframed[,1:(ddim - d + 1 - 1),, drop = FALSE]
    }
  }

  outcome <- list(reframed = reframed, head_list = head_list, tail_list = tail_list)
  return(outcome)
}

###
reframed_integration <- function(reframed, head_list)
{
  if(is.null(head_list)){return(reframed)}

  diff <- length(head_list)
  for(d in diff:1)
  {
    reframed <- abind(head_list[[d]], reframed, along = 2)
    reframed <- aperm(apply(reframed, - 2, cumsum), c(2, 1, 3))
  }

  return(reframed)
}


###
reframed_multiple_differentiation <- function(reframed, diff)
{
  if(length(diff)==1 || var(diff)==0)
  {
    model <- reframed_differentiation(reframed, diff[1])
    reframed <- model$reframed
    dmodels <- list(model)
    spare_parts <- NULL
  }

  if(length(diff) > 1 && var(diff) > 0)
  {
    reframed_list <- split(reframed, along = 3, drop = FALSE)
    dmodels <- map2(reframed_list, diff, ~ reframed_differentiation(.x, .y))
    dfix <- max(diff) - diff + 1
    difframed_list <- map2(dmodels, dfix, ~ .x$reframed[,.y:dim(.x$reframed)[2],,drop=FALSE])
    reframed <- abind(difframed_list, along = 3)
    spare_parts <- map2(dmodels, dfix, ~ if(.y > 1){.x$reframed[,1:(.y-1),,drop=FALSE]} else {NULL})
  }

  outcome <- list(reframed = reframed, dmodels = dmodels, spare_parts = spare_parts)
  return(outcome)
}

###
reframed_multiple_integration <- function(reframed, dmodels, spare_parts = NULL, pred = FALSE)
{
  if(length(dmodels)==1)
  {
    if(pred==FALSE){base_list <- dmodels[[1]]$head_list}
    if(pred==TRUE){base_list <- dmodels[[1]]$tail_list}
    reframed <- reframed_integration(reframed, base_list)
    if(pred==TRUE){reframed <- reframed[,(length(base_list)+1):dim(reframed)[2],, drop = FALSE]}
  }

  if(length(dmodels) > 1)
  {
    reframed_list <- split(reframed, along = 3, drop = FALSE)

    if(pred==FALSE)
    {
      base_lists <- map(dmodels, ~ .x$head_list)
      recon_list <- map2(spare_parts, reframed_list, ~ abind(.x, .y, along = 2))
      inframed_list <- map2(recon_list, base_lists, ~ reframed_integration(.x, .y))
    }

    if(pred==TRUE)
    {
      base_lists <- map(dmodels, ~ .x$tail_list)
      inframed_list <- map2(reframed_list, base_lists, ~ reframed_integration(.x, .y))
      dim_cols <- map_dbl(inframed_list, ~ dim(.x)[2])
      fix_cols <- dim_cols - min(dim_cols) + 1 ###FIXING FOR SPARE PARTS
      inframed_list <- pmap(list(inframed_list, fix_cols, dim_cols), ~ ..1[,..2:..3,,drop=FALSE])
      fix_cols2 <- min(map_dbl(base_lists, ~ length(.x)))+1 ###FIXING FOR DIFF
      inframed_list <- map(inframed_list, ~ .x[,fix_cols2:dim(.x)[2],,drop=FALSE])
    }

    reframed <- abind(inframed_list, along = 3)
  }

  return(reframed)
}


###
ts_graph <- function(x_hist, y_hist, x_forcat, y_forcat, lower = NULL, upper = NULL, line_size = 1.3, label_size = 11,
                     forcat_band = "darkorange", forcat_line = "darkorange", hist_line = "gray43",
                     label_x = "Horizon", label_y= "Forecasted Var", dbreak = NULL, date_format = "%b-%d-%Y")
{

  all_data <- data.frame(x_all = c(x_hist, x_forcat), y_all = c(y_hist, y_forcat))
  forcat_data <- data.frame(x_forcat = x_forcat, y_forcat = y_forcat)

  if(!is.null(lower) & !is.null(upper)){forcat_data$lower <- lower; forcat_data$upper <- upper}

  plot <- ggplot()+geom_line(data = all_data, aes_string(x = "x_all", y = "y_all"), color = hist_line, size = line_size)
  if(!is.null(lower) & !is.null(upper)){plot <- plot + geom_ribbon(data = forcat_data, aes_string(x = "x_forcat", ymin = "lower", ymax = "upper"), alpha = 0.3, fill = forcat_band)}
  plot <- plot + geom_line(data = forcat_data, aes_string(x = "x_forcat", y = "y_forcat"), color = forcat_line, size = line_size)
  if(!is.null(dbreak)){plot <- plot + scale_x_date(name = paste0("\n", label_x), date_breaks = dbreak, date_labels = date_format)}
  if(is.null(dbreak)){plot <- plot + xlab(label_x)}
  plot <- plot + scale_y_continuous(name = paste0(label_y, "\n"), labels = number)
  plot <- plot + ylab(label_y)  + theme_bw()
  plot <- plot + theme(axis.text=element_text(size=label_size), axis.title=element_text(size=label_size + 2))

  return(plot)
}

###
block_sampler <- function(data, seq_len, n_blocks, block_minset, stride)
{
  n_feat <- ncol(data)
  data_len <- nrow(data)

  if(floor(data_len/n_blocks) <= seq_len + block_minset - 1){stop("insufficient data for ", n_blocks," blocks\n")}
  if(n_blocks == 1){stop("only one block available for the sequence length\n")}

  block_index <- sort(c(rep(1:n_blocks, each = floor(data_len/n_blocks)), rep(1, data_len%%n_blocks)))

  data_list <- split(data, along = 1, subsets = block_index, drop = FALSE)
  block_set <- map(data_list, ~ block_reframer(.x, seq_len, stride))
  block_size <- map_dbl(block_set, ~ nrow(.x))
  if(any(block_size == 1)){stop("blocks with single sequence are not enough\n")}

  outcome <- list(block_set = block_set, block_index = block_index)

  return(outcome)
}

###
best_deriv <- function(ts, max_diff = 3, min_default = NULL, thresh = 0.001)
{
  pvalues <- vector(mode = "double", length = as.integer(max_diff))

  for(d in 1:(max_diff + 1))
  {
    model <- lm(ts ~ t, data.frame(ts, t = 1:length(ts)))
    pvalues[d] <- with(summary(model), pf(fstatistic[1], fstatistic[2], fstatistic[3],lower.tail=FALSE))
    ts <- diff(ts)
  }

  best <- tail(cumsum(pvalues < thresh), 1)
  if(is.numeric(min_default) && best < min_default){best <- min_default}

  return(best)
}

###
block_reframer <- function(df, seq_len, stride)
{
  reframe_list <- map(df, ~ smart_reframer(.x, seq_len, stride))
  reframed <- abind(reframe_list, along = 3)
  rownames(reframed) <- NULL
  return(reframed)
}

###
smart_reframer <- function(ts, seq_len, stride)
{
  n_length <- length(ts)
  if(seq_len > n_length | stride > n_length){stop("vector too short for sequence length or stride")}
  if(n_length%%seq_len > 0){ts <- tail(ts, - (n_length%%seq_len))}
  n_length <- length(ts)
  idx <- base::seq(from = 1, to = (n_length - seq_len + 1), by = 1)
  reframed <- t(sapply(idx, function(x) ts[x:(x+seq_len-1)]))
  if(seq_len == 1){reframed <- t(reframed)}
  idx <- rev(base::seq(nrow(reframed), 1, - stride))
  reframed <- reframed[idx,,drop = F]
  return(reframed)
}

###
qpred <- function(raw_pred, ts, ci, error_scale = "naive", error_benchmark = "naive")
{
  raw_pred <- doxa_filter(ts, raw_pred)
  quants <- sort(unique(c((1-ci)/2, 0.25, 0.5, 0.75, ci+(1-ci)/2)))

  p_stats <- function(x){c(min = suppressWarnings(min(x, na.rm = TRUE)), quantile(x, probs = quants, na.rm = TRUE), max = suppressWarnings(max(x, na.rm = TRUE)), mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE), mode = suppressWarnings(mlv1(x[is.finite(x)], method = "shorth")), kurtosis = suppressWarnings(kurtosis(x[is.finite(x)], na.rm = TRUE)), skewness = suppressWarnings(skewness(x[is.finite(x)], na.rm = TRUE)))}
  quant_pred <- as.data.frame(t(as.data.frame(apply(raw_pred, 2, p_stats))))
  p_value <- apply(raw_pred, 2, function(x) ecdf(x)(seq(min(raw_pred), max(raw_pred), length.out = 1000)))
  divergence <- c(max(p_value[,1] - seq(0, 1, length.out = 1000)), apply(p_value[,-1, drop = FALSE] - p_value[,-ncol(p_value), drop = FALSE], 2, function(x) abs(max(x, na.rm = TRUE))))
  upside_prob <- c(mean((raw_pred[,1]/tail(ts, 1)) > 1, na.rm = T), apply(apply(raw_pred[,-1, drop = FALSE]/raw_pred[,-ncol(raw_pred), drop = FALSE], 2, function(x) x > 1), 2, mean, na.rm = T))
  iqr_to_range <- (quant_pred[, "75%"] - quant_pred[, "25%"])/(quant_pred[, "max"] - quant_pred[, "min"])
  above_to_below_range <- (quant_pred[, "max"] - quant_pred[, "50%"])/(quant_pred[, "50%"] - quant_pred[, "min"])
  quant_pred <- round(cbind(quant_pred, iqr_to_range, above_to_below_range, upside_prob, divergence), 4)
  rownames(quant_pred) <- NULL

  return(quant_pred)
}

###
doxa_filter <- function(ts, mat)
{
  discrete_check <- all(ts%%1 == 0)
  all_positive_check <- all(ts >= 0)
  all_negative_check <- all(ts <= 0)
  monotonic_increase_check <- all(diff(ts) >= 0)
  monotonic_decrease_check <- all(diff(ts) <= 0)

  monotonic_fixer <- function(x, mode)
  {
    model <- recursive_diff(x, 1)
    vect <- model$vector
    if(mode == 0){vect[vect < 0] <- 0; vect <- invdiff(vect, model$head_value, add = TRUE)}
    if(mode == 1){vect[vect > 0] <- 0; vect <- invdiff(vect, model$head_value, add = TRUE)}
    return(vect)
  }

  if(all_positive_check){mat[mat < 0] <- 0}
  if(all_negative_check){mat[mat > 0] <- 0}
  if(discrete_check){mat <- round(mat)}
  if(monotonic_increase_check){mat <- t(apply(mat, 1, function(x) monotonic_fixer(x, mode = 0)))}
  if(monotonic_decrease_check){mat <- t(apply(mat, 1, function(x) monotonic_fixer(x, mode = 1)))}

  mat <- na.omit(mat)

  return(mat)
}


###
custom_metrics <- function(holdout, forecast, actuals, error_scale = "naive", error_benchmark = "naive")
{
    scale <- switch(error_scale, "deviation" = sd(actuals), "naive" = mean(abs(diff(actuals))))
    benchmark <- switch(error_benchmark, "average" = rep(mean(forecast), length(forecast)), "naive" = rep(tail(actuals, 1), length(forecast)))
    me <- ME(holdout, forecast, na.rm = TRUE)
    mae <- MAE(holdout, forecast, na.rm = TRUE)
    mse <- MSE(holdout, forecast, na.rm = TRUE)
    rmsse <- RMSSE(holdout, forecast, scale, na.rm = TRUE)
    mre <- MRE(holdout, forecast, na.rm = TRUE)
    mpe <- MPE(holdout, forecast, na.rm = TRUE)
    mape <- MAPE(holdout, forecast, na.rm = TRUE)
    rmae <- rMAE(holdout, forecast, benchmark, na.rm = TRUE)
    rrmse <- rRMSE(holdout, forecast, benchmark, na.rm = TRUE)
    rame <- rAME(holdout, forecast, benchmark, na.rm = TRUE)
    mase <- MASE(holdout, forecast, scale, na.rm = TRUE)
    smse <- sMSE(holdout, forecast, scale, na.rm = TRUE)
    sce <- sCE(holdout, forecast, scale, na.rm = TRUE)
    out <- round(c(me = me, mae = mae, mse = mse, rmsse = rmsse, mpe = mpe, mape = mape, rmae = rmae, rrmse = rrmse, rame = rame, mase = mase, smse = smse, sce = sce), 3)

  return(out)
}

###
plotter <- function(quant_pred, ci, ts, dates = NULL, time_unit = NULL, feat_name)
{

  seq_len <- nrow(quant_pred)
  n_ts <- length(ts)

  if(!is.null(dates) & !is.null(time_unit))
  {
    start <- as.Date(tail(dates, 1))
    new_dates<- seq.Date(from = start, length.out = seq_len, by = time_unit)
    x_hist <- dates
    x_forcat <- new_dates
    rownames(quant_pred) <- as.character(new_dates)
  }
  else
  {
    x_hist <- 1:n_ts
    x_forcat <- (n_ts + 1):(n_ts + seq_len)
    rownames(quant_pred) <- paste0("t", 1:seq_len)
  }

  quant_pred <- as.data.frame(quant_pred)
  x_lab <- paste0("Forecasting Horizon for sequence n = ", seq_len)
  y_lab <- paste0("Forecasting Values for ", feat_name)

  lower_b <- paste0((1-ci)/2 * 100, "%")
  upper_b <- paste0((ci+(1-ci)/2) * 100, "%")

  plot <- ts_graph(x_hist = x_hist, y_hist = ts, x_forcat = x_forcat, y_forcat = quant_pred[, "50%"], lower = quant_pred[, lower_b], upper = quant_pred[, upper_b], label_x = x_lab, label_y = y_lab)
  return(plot)
}

###
smart_head <- function(x, n)
{
  if(n != 0){return(head(x, n))}
  if(n == 0){return(x)}
}

###
smart_tail <- function(x, n)
{
  if(n != 0){return(tail(x, n))}
  if(n == 0){return(x)}
}

###
gap_fixer <- function(df, date_feat, verbose, omit)
{
  if(omit == TRUE)
  {
    df <- na.omit(df)
    if(!is.null(date_feat)){df[[date_feat]] <- as.Date(as.character(df[[date_feat]]))}
    return(df)
  }

  if(!is.null(date_feat))
  {
    dates <- as.Date(as.character(df[[date_feat]]))
    df[[date_feat]] <- dates
    main_freq <- mfv1(diff.Date(dates))
    fixed_dates <- data.frame(seq(head(dates, 1), tail(dates, 1), by = main_freq))
    colnames(fixed_dates) <- date_feat
    fixed_df <- suppressMessages(left_join(fixed_dates, df))
    df <- as.data.frame(map(fixed_df, ~ na_kalman(.x)))
    if(verbose == TRUE){cat("date and value gaps filled with kalman imputation\n")}
  }

  else
  {
    if(anyNA(df)){df <- as.data.frame(map(df, ~ na_kalman(.x))); if(verbose == TRUE){cat("value gaps filled with kalman imputation\n")}}
    else {return(df)}
  }

  return(df)
}

###
recursive_diff <- function(vector, deriv)
{
  vector <- unlist(vector)
  head_value <- vector("numeric", deriv)
  tail_value <- vector("numeric", deriv)
  if(deriv==0){head_value = NULL; tail_value = NULL}
  if(deriv > 0){for(i in 1:deriv){head_value[i] <- head(vector, 1); tail_value[i] <- tail(vector, 1); vector <- diff(vector)}}
  outcome <- list(vector = vector, head_value = head_value, tail_value = tail_value)
  return(outcome)
}

###
invdiff <- function(vector, heads, add = FALSE)
{
  vector <- unlist(vector)
  if(is.null(heads)){return(vector)}
  for(d in length(heads):1){vector <- cumsum(c(heads[d], vector))}
  if(add == FALSE){return(vector[-c(1:length(heads))])} else {return(vector)}
}





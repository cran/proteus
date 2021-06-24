#' proteus
#'
#' @param data A data frame with time features on columns and possibly a date column (not mandatory)
#' @param target Vector of strings. Names of the time features to be jointly analyzed
#' @param future Positive integer. The future dimension with number of time-steps to be predicted
#' @param past Positive integer. Length of past sequences
#' @param ci Positive numeric. Confidence interval. Default: 0.8
#' @param deriv Positive integer or vector. Number of recursive differentiation operations for each time feature: for example, c(2, 1, 3) means the first feature will be differentiated two times, the second only one, the third three times. Default: 1 for each time feature.
#' @param shift Vector of positive integers. Allow for target variables to shift ahead of time. Zero means no shift. Length must be equal to the number of targets. Default: 0.
#' @param smoother Logical. Perform optimal smoothing using standard loess for each time feature. Default: FALSE
#' @param t_embed Positive integer. Number of embedding for the temporal dimension. Minimum value is equal to 2. Default: 30.
#' @param activ String. Activation function to be used by the forward network. Implemented functions are: "linear", "leaky_relu", "celu", "elu", "gelu", "selu", "softplus", "bent", "snake", "softmax", "softmin", "softsign", "sigmoid", "tanh", "tanhshrink", "swish", "hardtanh", "mish". Default: "linear".
#' @param nodes Positive integer. Nodes for the forward neural net. Default: 32.
#' @param distr String. Distribution to be used by variational model. Implemented distributions are: "normal", "genbeta", "gev", "gpd", "genray", "cauchy", "exp", "logis", "chisq", "gumbel", "laplace", "lognorm". Default: "normal".
#' @param optim String. Optimization method. Implemented methods are: "adadelta", "adagrad", "rmsprop", "rprop", "sgd", "asgd", "adam".
#' @param loss_metric String. Loss function for the variational model. Two options: "elbo" or "crps". Default: "crps".
#' @param epochs Positive integer. Default: 30.
#' @param lr Positive numeric. Learning rate. Default: 0.01.
#' @param patience Positive integer. Waiting time (in epochs) before evaluating the overfit performance. Default: epochs.
#' @param verbose Logical. Default: TRUE
#' @param seed Random seed. Default: 42.
#' @param dev String. Torch implementation of computational platform: "cpu" or "cuda" (gpu). Default: "cpu".
#' @param dates Vector of strings. Vector with date strings for computing the prediction dates. Default: NULL (progressive numbers).
#' @param dbreak String. Minimum time marker for x-axis plot, in liberal form: i.e., "3 months", "1 week", "20 days". Default: NULL.
#' @param days_off String. Weekdays to exclude (i.e., c("saturday", "sunday")). Default: NULL.
#' @param rolling_blocks Logical. Option for incremental or rolling window. Default: FALSE.
#' @param n_blocks Positive integer. Number of distinct blocks for backtesting. Default: 4.
#' @param block_minset Positive integer. Minimum number of sequence to create a block. Default: 30.
#' @param batch_size Positive integer. Default: 30.
#' @param sequence_stride Logical. When FALSE, each sequence will be shifted of a single position in time; when TRUE, each sequence will be shifted for the full length of past + future (only distinct sequences allowed during reframing). Default: FALSE.
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#'@return This function returns a list including:
#' \itemize{
#'\item prediction: a table with quantile predictions, mean, std, mode, skewness and kurtosis for each time feature
#'\item plot: graph with history and prediction for each time feature
#'\item learning_error: train and test error for the joint time features (rmse, mae, mdae, mpe, mape, smape, rrse, rae)
#'\item feature_errors: train and test error for each time feature (rmse, mae, mdae, mpe, mape, smape, rrse, rae)
#'\item pred_stats: for each predicted time feature, IQR to range, Kullback-Leibler Divergence (compared to previous point in time), upside probability (compared to previous point in time). Average for all the prediction statics and comparison between the terminal and the first point in the prediction sequence.
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
#' @importFrom lubridate seconds_to_period
#' @importFrom scales number
#' @importFrom narray split
#' @importFrom stats lm median na.omit quantile
#' @importFrom utils head tail combn
#' @importFrom bizdays create.calendar add.bizdays bizseq
#'
#'@examples
#'\donttest{
#'proteus(amzn_aapl_fb, c("AMZN", "GOOGL", "FB"), future = 30, past = 100)
#'
#'proteus(amzn_aapl_fb, "AMZN", future = 30, past = 100, distr = "logis")
#'
#'proteus(amzn_aapl_fb, "AMZN", future = 30, past = 100, distr = "cauchy")
#'
#'proteus(amzn_aapl_fb, "AMZN", future = 30, past = 100, distr = "gev")
#'}
#'
proteus <- function(data, target, future, past, ci = 0.8, deriv = 1, shift = 0, smoother = FALSE,
                    t_embed = 30, activ = "linear", nodes = 32, distr = "normal", optim = "adam",
                    loss_metric = "crps", epochs = 30, lr = 0.01, patience = 10, verbose = TRUE,
                    seed = 42, dev ="cpu", dates = NULL, dbreak = NULL, days_off = NULL,
                    rolling_blocks = FALSE, n_blocks = 4, block_minset = 30, batch_size = 30, sequence_stride = FALSE)
{

  tic.clearlog()
  tic("proteus")

  ###PRECHECK
  if(max(deriv) >= future | max(deriv) >= past){stop("deriv cannot be equal or greater than future and/or past")}
  if(future <= 0 | past <= 0){stop("past and future must be strictly positive integers")}
  if(t_embed < 2){t_embed <- 2; if(verbose == TRUE){cat("setting t_embed to the minimum possible value (2)\n")}}
  if(n_blocks < 3){n_blocks <- 3; if(verbose == TRUE){cat("setting n_blocks to the minimum possible value (3)\n")}}


  ####PREPARATION
  set.seed(seed)
  torch_manual_seed(seed)

  data <- data[, target, drop = FALSE]
  n_feat <- ncol(data)

  ###MISSING IMPUTATION
  if(anyNA(data) & is.null(days_off)){data <- as.data.frame(map(data, ~ na_kalman(.x))); if(verbose == TRUE){cat("kalman imputation\n")}}
  if(anyNA(data) & !is.null(days_off)){data <- as.data.frame(na.omit(data)); if(verbose == TRUE){cat("omitting missings\n")}}
  n_length <- nrow(data)

  ###SHIFT
  if(any(shift > 0) & length(shift)==n_feat)
  {
    data <- map2(data, shift, ~ tail(.x, n_length - .y))
    n_length <- min(map_dbl(data, ~ length(.x)))
    data <- map_df(data, ~ head(.x, n_length))
  }

  ###SMOOTHING
  if(smoother==TRUE){data <- as.data.frame(map(data, ~ loess.as(x=1:n_length, y=.x)$fitted)); if(verbose == TRUE){cat("performing optimal smoothing\n")}}

  ###SEGMENTATION
  block_set <- block_sampler(data, seq_len = past + future, n_blocks, block_minset, sequence_stride)
  feature_block <- 1:past
  target_block <- (past - max(deriv) + 1):(past + future)

  block_features_errors <- vector("list", n_blocks-1)
  block_learning_errors <- vector("list", n_blocks-1)
  block_raw_errors <- vector("list", n_blocks-1)

  for(n in 1:(n_blocks-1))
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
    new_reframed <- reframe(new_data, past, sequence_stride)

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

    pred_train <- pred_fun(model, x_train, "mean", dev=dev)
    pred_test <- pred_fun(model, x_test, "mean", dev=dev)
    train_errors <- eval_metrics(y_train, pred_train)
    test_errors <- eval_metrics(y_test, pred_test)
    learning_error <- Reduce(rbind, list(train_errors, test_errors))
    rownames(learning_error) <- c("train", "test")
    block_learning_errors[[n]] <- learning_error

    ###INTEGRATION
    pred_train <- reframed_multiple_integration(pred_train, x_train_deriv_model$dmodels, pred = TRUE)
    pred_test <- reframed_multiple_integration(pred_test, x_test_deriv_model$dmodels, pred = TRUE)

    ###ERRORS
    predict_block <- (past + 1):(past + future)
    train_true <- train_reframed[,predict_block,,drop=FALSE]
    train_errors <- map2(smart_split(train_true, along = 3), smart_split(pred_train, along = 3), ~ eval_metrics(.x, .y))

    predict_block <- (past + 1):(past + future)
    test_true <- test_reframed[,predict_block,,drop=FALSE]
    test_errors <- map2(smart_split(test_true, along = 3), smart_split(pred_test, along = 3), ~ eval_metrics(.x, .y))
    block_raw_errors[[n]] <- aperm(apply(test_true - pred_test, c(1, 2, 3), mean, na.rm=TRUE), c(2, 1, 3))

    features_errors <- transpose(list(train_errors, test_errors))
    features_errors <- map(features_errors, ~ Reduce(rbind, .x))
    features_errors <- map(features_errors, ~ {rownames(.x) <- c("train", "test"); return(.x)})
    names(features_errors) <- target

    block_features_errors[[n]] <- features_errors
  }

  features_errors <- map(transpose(block_features_errors), ~ Reduce('+', .x)/(n_blocks-1))
  learning_error <- Reduce('+', block_learning_errors)/(n_blocks-1)
  raw_errors <- aperm(abind(block_raw_errors, along = 2), c(2, 1, 3))

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

  new_data_deriv_model <- reframed_multiple_differentiation(new_reframed, deriv)
  new_reframed <- new_data_deriv_model$reframed

  model <- nn_variational_model(target_len = future, seq_len = past - max(deriv), n_feat = n_feat, t_embed = t_embed, activ = activ, nodes = nodes, dev = dev, distr = distr)
  training <- training_function(model, x_train, y_train, x_test = NULL, y_test = NULL, loss_metric, optim, lr, epochs, patience, verbose, batch_size, distr, dev)
  train_history <- training$train_history
  model <- training$model

  ###NEW PREDICTION
  pred_new <- replicate(dim(raw_errors)[1], pred_fun(model, new_reframed, "sample", dev=dev), simplify = FALSE)
  pred_new <- map(pred_new, ~ reframed_multiple_integration(.x, new_data_deriv_model$dmodels, pred = TRUE))###UNLIST NEEDED ONLY HERE

  quants <- sort(unique(c((1-ci)/2, 0.25, 0.5, 0.75, ci+(1-ci)/2)))
  q_names <- paste0("q", quants * 100)
  stat_fun <- function(x) {round(c(min(x, na.rm=TRUE), quantile(x, probs = quants, na.rm = TRUE), max(x, na.rm=TRUE), mean(x, na.rm=TRUE), sd(x, na.rm=TRUE), mlv1(x, method = "shorth", na.rm = TRUE), skewness(x, na.rm=TRUE), kurtosis(x, na.rm=TRUE)), 3)}

  if(any(is.infinite(raw_errors))){raw_errors <- abind(map(smart_split(raw_errors, along = 3), ~ {.x[.x == Inf] <- max(.x[is.finite(.x)]); .x[.x== -Inf] <- min(.x[is.finite(.x)]); return(.x)}), along = 3)}
  integrated_pred <- abind(pred_new, along=1) + raw_errors
  pred_quantiles <- aperm(apply(integrated_pred, c(2, 3), stat_fun), c(2, 1, 3))

  prediction <- map(smart_split(pred_quantiles, along = 3), ~{rownames(.x) <- paste0("t", 1:future); colnames(.x) <- c("min", q_names, "max", "mean", "sd", "mode", "skewness", "kurtosis"); return(.x)})
  names(prediction) <- target

  ###PREDICTION STATISTICS
  avg_iqr_to_range <- round(map_dbl(prediction, ~ mean((.x[,"q75"] - .x[,"q25"])/(.x[,"max"] - .x[,"min"]))), 3)
  last_to_first_iqr <- round(map_dbl(prediction, ~ (.x[future,"q75"] - .x[future,"q25"])/(.x[1,"q75"] - .x[1,"q25"])), 3)
  kld_stats <- map(split(integrated_pred, along=3), ~ tryCatch(sequential_kld(t(.x)), error = function(e) NULL))
  if(!is.null(kld_stats)){kld_stats <- map_df(transpose(kld_stats), ~ unlist(.x))}
  upp_stats <- map(split(integrated_pred, along=3), ~ tryCatch(upside_probability(t(.x)), error = function(e) NULL))
  if(!is.null(upp_stats)){upp_stats <- map_df(transpose(upp_stats), ~ unlist(.x))}

  pred_stats <- as.data.frame(rbind(avg_iqr_to_range, last_to_first_iqr, kld_stats, upp_stats))###WRAPPED AS DF TO AVOID TIBBLE WARNING IN FOLLOWING LINE
  if(!is.null(kld_stats) & !is.null(upp_stats)){rownames(pred_stats) <- c("avg_iqr_to_range", "terminal_iqr_ratio", "avg_kl_divergence", "terminal_kl_divergence", "avg_upside_prob", "terminal_upside_prob")}
  if(!is.null(kld_stats) & is.null(upp_stats)){rownames(pred_stats) <- c("avg_iqr_to_range", "terminal_iqr_ratio", "avg_kl_divergence", "terminal_kl_divergence")}
  if(is.null(kld_stats) & !is.null(upp_stats)){rownames(pred_stats) <- c("avg_iqr_to_range", "terminal_iqr_ratio", "avg_upside_prob", "terminal_upside_prob")}
  if(is.null(kld_stats) & is.null(upp_stats)){rownames(pred_stats) <- c("avg_iqr_to_range", "terminal_iqr_ratio")}

  ###SETTING DATES
  if(!is.null(dates))
  {
    if(length(shift) != n_feat){shift <- rep(shift[1], n_feat)}
    start_date <- map(shift, ~ as.Date(tail(dates, 1)) + .x + 1)
    mycal <- create.calendar(name="mycal", weekdays=days_off)
    end_day <- map(start_date, ~ add.bizdays(.x, future, cal=mycal))
    pred_dates <- map2(start_date, end_day, ~ tail(bizseq(.x, .y, cal=mycal), future))
    prediction <- map2(prediction, pred_dates, ~ as.data.frame(cbind(dates=.y, .x)))
    prediction <- map(prediction, ~ {.x$dates <- as.Date(.x$dates, origin = "1970-01-01"); return(.x)})
    dates <- map(shift, ~ seq.Date(as.Date(head(dates, 1)) + .x, as.Date(tail(dates, 1)) + .x, length.out = n_length))
  }

  if(is.null(dates))
  {
    dates <- 1:n_length
    dates <- replicate(n_feat, dates, simplify = FALSE)
    pred_dates <- (n_length+1):(n_length+future)
    pred_dates <- replicate(n_feat, pred_dates, simplify = FALSE)
  }

  ###PREDICTION PLOT
  lower_name <- paste0("q", ((1-ci)/2) * 100)
  upper_name <- paste0("q", (ci+(1-ci)/2) * 100)

  plot <- pmap(list(data, prediction, target, dates, pred_dates), ~ ts_graph(x_hist = ..4, y_hist = ..1, x_forcat = ..5, y_forcat = ..2[, "mean"],
                                                                             lower = ..2[, lower_name], upper = ..2[, upper_name], label_x = paste0("Seq2Seq Variational Model (past = ", past ,", future = ", future,")"),
                                                                             label_y = paste0(str_to_title(..3), " Values"), dbreak = dbreak))

  toc(log = TRUE)
  time_log<-seconds_to_period(round(parse_number(unlist(tic.log())), 0))

  n_tensors <- length(model$parameters)
  n_parameters <- sum(map_dbl(model$parameters, ~ length(as.vector(as_array(.x)))))
  if(verbose==TRUE){cat("\nvariational model based on", distr, "latent distribution with", n_tensors, "tensors and", n_parameters, "parameters\n")}
  model_descr <- paste0("variational model based on ", distr, " latent distribution with ", n_tensors, " tensors and ", n_parameters, " parameters")

  ###OUTCOMES
  outcome <- list(model_descr = model_descr, prediction = prediction, plot = plot, learning_error = learning_error,
                  features_errors = features_errors, pred_stats = pred_stats, time_log = time_log)

  return(outcome)

}

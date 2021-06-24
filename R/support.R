#' support functions for proteus
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @import purrr
#' @import abind
#' @import torch
#' @import ggplot2
#' @import tictoc
#' @import readr
#' @import stringr
#' @import lubridate
#' @importFrom scales number
#' @importFrom narray split
#' @importFrom stats lm median na.omit quantile density rcauchy rchisq rexp rlnorm rlogis rnorm dcauchy dchisq dexp dlnorm dlogis dnorm pcauchy pchisq pexp plnorm plogis pnorm runif sd var
#' @importFrom utils head tail
#' @importFrom VGAM rgev pgev dgev rgpd pgpd dgpd rlaplace plaplace dlaplace rgenray pgenray dgenray rgumbel dgumbel pgumbel
#' @importFrom actuar rgenbeta pgenbeta dgenbeta
#' @importFrom modeest mlv1
#' @importFrom moments skewness kurtosis
##@importFrom torch nn_module


###SUPPORT

globalVariables(c("train_loss", "val_loss", "x_all", "y_all", "pgenray", "pgumbel", "plaplace", "dgenray", "dgumbel", "dlaplace", "dev"))

nn_mish <- nn_module(
  "nn_mish",
  initialize = function(beta = 1) {self$softplus <- nn_softplus(beta = beta)},
  forward = function(x) {x * torch_tanh(self$softplus(x))})

nn_fts <- nn_module(
  "nn_fts",
  initialize = function(T = - 0.001) {self$T <- nn_buffer(T)},
  forward = function(x) {nnf_leaky_relu(x) * torch_sigmoid(x) + T})

nn_snake <- nn_module(
  "nn_snake",
  initialize = function(a = 0.5) {self$a <- nn_buffer(a)},
  forward = function(x) {x + torch_square(torch_sin(self$a*x)/self$a)})

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
  initialize = function(act, dim = 2, alpha = 1, beta = 1, lambda = 0.5, min_val = -1, max_val = 1, a = 0.5)
  {
    if(act == "linear"){self$activ <- nn_identity()}
    if(act == "mish"){self$activ <- nn_mish(beta)}
    if(act == "leaky_relu"){self$activ <- nn_leaky_relu()}
    if(act == "celu"){self$activ <- nn_celu(alpha)}
    if(act == "elu"){self$activ <- nn_elu(alpha)}
    if(act == "gelu"){self$activ <- nn_gelu()}
    if(act == "selu"){self$activ <- nn_selu()}
    if(act == "softplus"){self$activ <- nn_softplus(beta)}
    if(act == "bent"){self$activ <- nn_bent()}
    if(act == "snake"){self$activ <- nn_snake(a)}
    if(act == "softmax"){self$activ <- nn_softmax(dim)}
    if(act == "softmin"){self$activ <- nn_softmin(dim)}
    if(act == "softsign"){self$activ <- nn_softsign()}
    if(act == "sigmoid"){self$activ <- nn_sigmoid()}
    if(act == "tanh"){self$activ <- nn_tanh()}
    if(act == "hardsigmoid"){self$activ <- nn_hardsigmoid()}
    if(act == "swish"){self$activ <- nn_swish()}
    if(act == "fts"){self$activ <- nn_fts()}
    if(act == "hardtanh"){self$activ <- nn_hardtanh(min_val, max_val)}
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
  },
  forward = function(x)
  {
    distr <- self$distr
    result <- self$var_model(x)
    outcome <- list(latent = result[[1]], params = result[-1], distr = distr)
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
  error <- as_array(latent$cpu()) - as_array(actual$cpu())
  heaviside_step <- error >= 0
  loss <- mean((latent_cdf - heaviside_step)^2)###MEAN INSTEAD OF SUM

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
pred_fun <- function(model, new_data, type = "sample", quant = 0.5, seed = as.numeric(Sys.time()), dev)
{
  torch_device(dev)
  if(!("torch_tensor" %in% class(new_data))){new_data <- torch_tensor(as.array(new_data))}
  if(type=="sample"){pred <- as_array(model(new_data)$latent$cpu())}

  if(type == "quant" | type == "mean" | type == "mode")
  {
    pred <- abind(replicate(100, as_array(model(new_data)$latent$cpu()), simplify = FALSE), along=-1)###DRAWING 100 SAMPLES FROM LATENT
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
eval_metrics <- function(actual, predicted)
{
  actual <- unlist(actual)
  predicted <- unlist(predicted)
  if(length(actual) != length(predicted)){stop("different lengths")}

  rmse <- sqrt(mean((actual - predicted)^2, na.rm = TRUE))
  mae <- mean(abs(actual - predicted), na.rm = TRUE)
  mdae <- median(abs(actual - predicted), na.rm = TRUE)
  mpe <- mean((actual - predicted)/actual, na.rm = TRUE)
  mape <- mean(abs(actual - predicted)/abs(actual), na.rm = TRUE)
  smape <- mean(abs(actual - predicted)/mean(c(abs(actual), abs(predicted))), na.rm = TRUE)
  rrse <- sqrt(sum((actual - predicted)^2, na.rm = TRUE))/sqrt(sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE))
  rae <- sum(abs(actual - predicted), na.rm = TRUE)/sum(abs(actual - mean(actual, na.rm = TRUE)), na.rm = TRUE)

  metrics <- round(c(rmse = rmse, mae = mae, mdae = mdae, mpe = mpe, mape = mape, smape = smape, rrse = rrse, rae = rae), 4)
  return(metrics)
}

###
sequential_kld <- function(matrix)
{
  n <- nrow(matrix)
  if(n == 1){return(NA)}
  dens <- apply(matrix, 1, function(x) density(x, from = min(matrix), to = max(matrix)))
  backward <- dens[-n]
  forward <- dens[-1]

  seq_kld <- map2_dbl(forward, backward, ~ sum(.x$y * log(.x$y/.y$y)))
  avg_seq_kld <- round(mean(seq_kld[is.finite(seq_kld)]), 3)

  ratios <- log(dens[[n]]$y/dens[[1]]$y)
  finite_index <- is.finite(ratios)

  end_to_end_kld <- dens[[n]]$y * log(dens[[n]]$y/dens[[1]]$y)
  end_to_end_kld <- round(sum(end_to_end_kld[finite_index]), 3)
  kld_stats <- rbind(avg_seq_kld, end_to_end_kld)

  return(kld_stats)
}

###
upside_probability <- function(matrix)
{
  n <- nrow(matrix)
  if(n == 1){return(NA)}
  growths <- matrix[-1,]/matrix[-n,] - 1
  dens <- apply(growths, 1, function(x) density(x, from = min(x), to = max(x)))
  avg_upp <- round(mean(map_dbl(dens, ~ sum(.x$y[.x$x>0])/sum(.x$y))), 3)
  end_growth <- matrix[n,]/matrix[1,] - 1
  end_to_end_dens <- density(end_growth, from = min(end_growth), to = max(end_growth))
  last_to_first_upp <- round(sum(end_to_end_dens$y[end_to_end_dens$x>0])/sum(end_to_end_dens$y), 3)
  upp_stats <- rbind(avg_upp, last_to_first_upp)
  return(upp_stats)
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
reframe<-function(df, length, sequence_stride = FALSE)
{
  if(sequence_stride == FALSE)
  {
    slice_list <- split(df, along=2)
    reframed <- abind(map(slice_list, ~ t(apply(embed(.x, dimension=length), 1, rev))), along=3)
  }

  if(sequence_stride == TRUE)
  {
    n_length <- nrow(df)
    n_seq <- floor(n_length/length)
    if(n_seq == 0){n_seq <- 1}
    seq_index <- rep(1:n_seq, each = length)
    df <- tail(df, length(seq_index))
    slice_list <- split(df, along=2)
    reframed <- map(slice_list, ~ Reduce(rbind, split(.x, along = 1, subsets = seq_index)))
    if(n_seq == 1){reframed <- map(reframed, ~ array(.x, dim = c(1, length, 1)))}
    reframed <- abind(reframed, along = 3)
    rownames(reframed) <- NULL
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

  plot <- ggplot()+geom_line(data = all_data, aes(x = x_all, y = y_all), color = hist_line, size = line_size)
  if(!is.null(lower) & !is.null(upper)){plot <- plot + geom_ribbon(data = forcat_data, aes(x = x_forcat, ymin = lower, ymax = upper), alpha = 0.3, fill = forcat_band)}
  plot <- plot + geom_line(data = forcat_data, aes(x = x_forcat, y = y_forcat), color = forcat_line, size = line_size)
  if(!is.null(dbreak)){plot <- plot + scale_x_date(name = paste0("\n", label_x), date_breaks = dbreak, date_labels = date_format)}
  if(is.null(dbreak)){plot <- plot + xlab(label_x)}
  plot <- plot + scale_y_continuous(name = paste0(label_y, "\n"), labels = number)
  plot <- plot + ylab(label_y)  + theme_bw()
  plot <- plot + theme(axis.text=element_text(size=label_size), axis.title=element_text(size=label_size + 2))

  return(plot)
}

###
block_sampler <- function(data, seq_len, n_blocks, block_minset, sequence_stride)
{
  n_feat <- ncol(data)
  data_len <- nrow(data)

  if(floor(data_len/n_blocks) <= seq_len + block_minset - 1){stop("insufficient data for ", n_blocks," blocks\n")}
  if(n_blocks == 1){stop("only one block available for the sequence length\n")}

  block_index <- sort(c(rep(1:n_blocks, each = floor(data_len/n_blocks)), rep(1, data_len%%n_blocks)))

  data_list <- split(data, along = 1, subsets = block_index, drop = FALSE)
  block_set <- map(data_list, ~ reframe(.x, seq_len, sequence_stride))
  block_size <- map_dbl(block_set, ~ nrow(.x))
  if(any(block_size == 1)){stop("blocks with single sequence are not enough\n")}

  return(block_set)
}

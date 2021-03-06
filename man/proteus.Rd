% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/main.R, R/proteus.R
\docType{package}
\name{proteus}
\alias{proteus}
\alias{proteus-package}
\title{proteus}
\usage{
proteus(
  data,
  target,
  future,
  past,
  ci = 0.8,
  deriv = 1,
  shift = 0,
  smoother = FALSE,
  t_embed = 30,
  activ = "linear",
  nodes = 32,
  distr = "normal",
  optim = "adam",
  loss_metric = "crps",
  epochs = 30,
  lr = 0.01,
  patience = 10,
  verbose = TRUE,
  seed = 42,
  dev = "cpu",
  dates = NULL,
  dbreak = NULL,
  days_off = NULL,
  rolling_blocks = FALSE,
  n_blocks = 4,
  block_minset = 30,
  batch_size = 30,
  sequence_stride = FALSE
)
}
\arguments{
\item{data}{A data frame with time features on columns and possibly a date column (not mandatory)}

\item{target}{Vector of strings. Names of the time features to be jointly analyzed}

\item{future}{Positive integer. The future dimension with number of time-steps to be predicted}

\item{past}{Positive integer. Length of past sequences}

\item{ci}{Positive numeric. Confidence interval. Default: 0.8}

\item{deriv}{Positive integer or vector. Number of recursive differentiation operations for each time feature: for example, c(2, 1, 3) means the first feature will be differentiated two times, the second only one, the third three times. Default: 1 for each time feature.}

\item{shift}{Vector of positive integers. Allow for target variables to shift ahead of time. Zero means no shift. Length must be equal to the number of targets. Default: 0.}

\item{smoother}{Logical. Perform optimal smoothing using standard loess for each time feature. Default: FALSE}

\item{t_embed}{Positive integer. Number of embedding for the temporal dimension. Minimum value is equal to 2. Default: 30.}

\item{activ}{String. Activation function to be used by the forward network. Implemented functions are: "linear", "leaky_relu", "celu", "elu", "gelu", "selu", "softplus", "bent", "snake", "softmax", "softmin", "softsign", "sigmoid", "tanh", "tanhshrink", "swish", "hardtanh", "mish". Default: "linear".}

\item{nodes}{Positive integer. Nodes for the forward neural net. Default: 32.}

\item{distr}{String. Distribution to be used by variational model. Implemented distributions are: "normal", "genbeta", "gev", "gpd", "genray", "cauchy", "exp", "logis", "chisq", "gumbel", "laplace", "lognorm". Default: "normal".}

\item{optim}{String. Optimization method. Implemented methods are: "adadelta", "adagrad", "rmsprop", "rprop", "sgd", "asgd", "adam".}

\item{loss_metric}{String. Loss function for the variational model. Two options: "elbo" or "crps". Default: "crps".}

\item{epochs}{Positive integer. Default: 30.}

\item{lr}{Positive numeric. Learning rate. Default: 0.01.}

\item{patience}{Positive integer. Waiting time (in epochs) before evaluating the overfit performance. Default: epochs.}

\item{verbose}{Logical. Default: TRUE}

\item{seed}{Random seed. Default: 42.}

\item{dev}{String. Torch implementation of computational platform: "cpu" or "cuda" (gpu). Default: "cpu".}

\item{dates}{Vector of strings. Vector with date strings for computing the prediction dates. Default: NULL (progressive numbers).}

\item{dbreak}{String. Minimum time marker for x-axis plot, in liberal form: i.e., "3 months", "1 week", "20 days". Default: NULL.}

\item{days_off}{String. Weekdays to exclude (i.e., c("saturday", "sunday")). Default: NULL.}

\item{rolling_blocks}{Logical. Option for incremental or rolling window. Default: FALSE.}

\item{n_blocks}{Positive integer. Number of distinct blocks for backtesting. Default: 4.}

\item{block_minset}{Positive integer. Minimum number of sequence to create a block. Default: 30.}

\item{batch_size}{Positive integer. Default: 30.}

\item{sequence_stride}{Logical. When FALSE, each sequence will be shifted of a single position in time; when TRUE, each sequence will be shifted for the full length of past + future (only distinct sequences allowed during reframing). Default: FALSE.}
}
\value{
This function returns a list including:
\itemize{
\item prediction: a table with quantile predictions, mean, std, mode, skewness and kurtosis for each time feature
\item plot: graph with history and prediction for each time feature
\item learning_error: train and test error for the joint time features (rmse, mae, mdae, mpe, mape, smape, rrse, rae)
\item feature_errors: train and test error for each time feature (rmse, mae, mdae, mpe, mape, smape, rrse, rae)
\item pred_stats: for each predicted time feature, IQR to range, Kullback-Leibler Divergence (compared to previous point in time), upside probability (compared to previous point in time). Average for all the prediction statics and comparison between the terminal and the first point in the prediction sequence.
\item time_log: computation time.
}
}
\description{
Seq2seq time-feature analysis based on variational model, with a wide range of distributions available for the latent variable.
}
\examples{
\donttest{
proteus(amzn_aapl_fb, c("AMZN", "GOOGL", "FB"), future = 30, past = 100)

proteus(amzn_aapl_fb, "AMZN", future = 30, past = 100, distr = "logis")

proteus(amzn_aapl_fb, "AMZN", future = 30, past = 100, distr = "cauchy")

proteus(amzn_aapl_fb, "AMZN", future = 30, past = 100, distr = "gev")
}

}
\seealso{
Useful links:
\itemize{
  \item \url{https://rpubs.com/giancarlo_vercellino/proteus}
}

}
\author{
Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
}

library(readr)
library(kknn)
library(ggplot2)
library(data.table)
library(glmnet)

### Question 1

# read in data
optdigits <- readr::read_csv("optdigits.csv", col_names = FALSE, col_types = cols())

# treat Y variable "X65" as factor
optdigits$X65 <- as.factor(optdigits$X65)

# create train/validation/test partitions
all_indices <- 1:(nrow(optdigits))
set.seed(12345)
train_indices <- sample(all_indices, ceiling(0.5*nrow(optdigits)))
val_test_indices <- all_indices[!(all_indices %in% train_indices)]
set.seed(12345)
val_indices <- sample(val_test_indices, ceiling(0.5*length(val_test_indices))) # 25% of all_indices == 50% of val_test_indices
test_indices <- val_test_indices[!(val_test_indices %in% val_indices)]
train_data <- optdigits[train_indices, ]
val_data <- optdigits[val_indices, ]
test_data <- optdigits[test_indices, ]

# train k-NN model with k = 30 using kknn function as asked
model_30_NN_train <- kknn(formula = X65~., kernel = "rectangular", train = train_data, test = train_data, k = 30)
model_30_NN_test <- kknn(formula = X65~., kernel = "rectangular", train = train_data, test = test_data, k = 30)
model_30_NN_val <- kknn(formula = X65~., kernel = "rectangular", train = train_data, test = val_data, k = 30)

# Confusion matrices for train and test datasets
train_c_table <- table(train_data$X65, fitted(model_30_NN_train))
val_c_table <- table(val_data$X65, fitted(model_30_NN_val))
test_c_table <- table(test_data$X65, fitted(model_30_NN_test))

cat("Confusion matrix for Training Dataset is: \n")
print(train_c_table)
cat("Confusion matrix for Validation Dataset is: \n")
print(val_c_table)
cat("Confusion matrix for Test Dataset is: \n")
print(test_c_table)

# misclassification errors for the training and test data
training_misclass <- round((1-sum(diag(train_c_table))/sum(train_c_table))*100, 2)
val_misclass <- round((1-sum(diag(val_c_table))/sum(val_c_table))*100, 2)
test_misclass <- round((1-sum(diag(test_c_table))/sum(test_c_table))*100, 2)

cat("Training mis-classification rate is: ", training_misclass, "%", sep = "")
cat("Validation mis-classification rate is: ", val_misclass, "%", sep = "")
cat("Test mis-classification rate is: ", test_misclass, "%", sep = "")

# cases with (a) 'ground truth' class = '8', and (b) highest or lowest modelled probability of belonging to class '8'
class_8_probs_for_true_8 <- model_30_NN_train$prob[train_data$X65 == 8, 9]
class_8_indices <- which(train_data$X65 == 8)
sorted_prob_indices <- order(class_8_probs_for_true_8)
top_1_prob_data <- unlist(train_data[class_8_indices[sorted_prob_indices[length(sorted_prob_indices)]], 1:64])
top_2_prob_data <- unlist(train_data[class_8_indices[sorted_prob_indices[length(sorted_prob_indices) - 1]], 1:64])
bott_1_prob_data <- unlist(train_data[class_8_indices[sorted_prob_indices[1]], 1:64])
bott_2_prob_data <- unlist(train_data[class_8_indices[sorted_prob_indices[2]], 1:64])
bott_3_prob_data <- unlist(train_data[class_8_indices[sorted_prob_indices[3]], 1:64])

# transform 1x64 into 8x8
top_1_prob_data <- matrix(top_1_prob_data, 8, 8, TRUE)
top_2_prob_data <- matrix(top_2_prob_data, 8, 8, TRUE)
bott_1_prob_data <- matrix(bott_1_prob_data, 8, 8, TRUE)
bott_2_prob_data <- matrix(bott_2_prob_data, 8, 8, TRUE)
bott_3_prob_data <- matrix(bott_3_prob_data, 8, 8, TRUE)

# visualise
heatmap(top_1_prob_data, Colv=NA, Rowv=NA, col = c("white", "black"))
heatmap(top_2_prob_data, Colv=NA, Rowv=NA, col = c("white", "black"))
heatmap(bott_1_prob_data, Colv=NA, Rowv=NA, col = c("white", "black"))
heatmap(bott_2_prob_data, Colv=NA, Rowv=NA, col = c("white", "black"))
heatmap(bott_3_prob_data, Colv=NA, Rowv=NA, col = c("white", "black"))

# optimise hyper-parameter 'k' based on validation performance measured by (a) misclassification rate (b) cross-entropy
missclass_train <- c()
missclass_val <- c()
cross_entropy_train <- c()
cross_entropy_val <- c()
num_classes <- length(unique(train_data$X65))
num_train_examples <- nrow(train_data)
num_val_examples <- nrow(val_data)
one_hot_y_train <- t(sapply(as.numeric(train_data$X65), 
                            function(x) {c(rep(0, x - 1), 1, rep(0, num_classes - x))}))
one_hot_y_val <- t(sapply(as.numeric(val_data$X65), 
                          function(x) {c(rep(0, x - 1), 1, rep(0, num_classes - x))}))

for (k_choice in 1:30) {
  
  # fit model
  train_model <- kknn(formula = X65~., kernel = "rectangular", train = train_data, test = train_data, k = k_choice)
  val_model <- kknn(formula = X65~., kernel = "rectangular", train = train_data, test = val_data, k = k_choice)
  
  # contingency tables
  train_c_table <- table(train_data$X65, fitted(train_model))
  val_c_table <- table(val_data$X65, fitted(val_model))
  
  # misclassification rates
  missclass_train <- c(missclass_train, 1-sum(diag(train_c_table))/sum(train_c_table))
  missclass_val <- c(missclass_val, 1-sum(diag(val_c_table))/sum(val_c_table))
  
  # cross-entropy
  cross_entropy_train <- c(cross_entropy_train, sum(one_hot_y_train * -log(train_model$prob + 10^-15))) #/num_train_examples
  cross_entropy_val <- c(cross_entropy_val, sum(one_hot_y_val * -log(val_model$prob + 10^-15))) #/num_val_examples
  
}

missclass <- melt(data.table(k = 1:30, Training = missclass_train, Validation = missclass_val), "k",
                  variable.name = "Legend")
cross_entropy <- melt(data.table(k = 1:30, Training = cross_entropy_train, Validation = cross_entropy_val),
                      "k", variable.name = "Legend")

# plot misclassification rates and cross-entropy by 'k' for training and validation datasets
ggplot(missclass) + geom_line(aes(k, value, colour = Legend)) + theme_bw() + 
  geom_vline(xintercept = 4, linetype = "dotted") + scale_x_continuous(breaks = 1:30) +
  xlab("Hyper-Parameter 'k'") + ylab("Mis-classification Rates") + 
  ggtitle("Finding the Optimal Hyper-Parameter")

ggplot(cross_entropy) + geom_line(aes(k, value, colour = Legend)) + theme_bw() + 
  geom_vline(xintercept = 6, linetype = "dotted") + scale_x_continuous(breaks = 1:30) +
  xlab("Hyper-Parameter 'k'") + ylab("Cross-Entropy") + 
  ggtitle("Finding the Optimal Hyper-Parameter")

### Question 2

# read in data
parkinsons <- read_csv("parkinsons.csv", col_types = cols())

# drop columns not to be used for prediction
parkinsons <- parkinsons[, -c(1, 2, 3, 4, 6)]

# bayesian model

## $$\mathbf{y} | \mathbb{X}, w_0, \mathbf{w} \sim N(w_0 + \mathbb{X}\mathbf{w}, \sigma^2\cdot \mathbb{I})$$
## $$\mathbf{w} \sim N(0, \sigma^2/\lambda \cdot \mathbb{I})$$

# create train/test partitions
all_indices <- 1:(nrow(parkinsons))
set.seed(12345)
train_indices <- sample(all_indices, ceiling(0.6*nrow(parkinsons)))
test_indices <- all_indices[!(all_indices %in% train_indices)]
train_data <- parkinsons[train_indices, ]
test_data <- parkinsons[test_indices, ]

# scale data -- see https://sebastianraschka.com/faq/docs/scale-training-test.html for why training mu/sigma are used for scaling test data
n_col <- ncol(parkinsons)
train_mu <- sapply(1:n_col, function(x) mean(unlist(train_data[x])))
train_sigma <- sapply(1:n_col, function(x) sd(unlist(train_data[x])))

train_data <- as.matrix(as.data.frame(
  sapply(1:n_col, function(x) (train_data[x] - train_mu[x])/train_sigma[x]))) # as.matrix has no effect without as.data.frame!!!
test_data <- as.matrix(as.data.frame(
  sapply(1:n_col, function(x) (test_data[x] - train_mu[x])/train_sigma[x])))

# set up helper functions

model_log_likelihood <- function(w_vector, sigma, X) { # built for no intercept models, as asked
  
  n <- nrow(X)
  term1 <- -n/2*log(2*pi*sigma^2)
  term2 <- -sum(sapply(1:n, function(x) X[x, 1] - X[x, -1] %*% w_vector)^2)/2/sigma^2 # assumes y = X[, 1]
  
  return(term1 + term2)
  
}

ridge <- function(w_vector_sigma, X, lambda) { # built for no intercept models, as asked
  
  n <- ncol(X)
  sigma <- w_vector_sigma[n]
  w_vector <- w_vector_sigma[1:(n-1)]
  return(lambda*sum(w_vector^2) - model_log_likelihood(w_vector, sigma, X))
  
}

ridge_opt <- function(X, lambda) {
  
  n <- ncol(X)
  par <- rnorm(n-1) # initial parameter values (to seed optimisation); random
  par[n] <- 1 # initialise with positive value
  return(optim(par = par, fn = ridge, method = "BFGS", X = X, lambda = lambda)$par)
  
}

degrees_of_freedom <- function(X, num_samples, num_simulations, optimal_weigths_sigma) {
  
  n <- ncol(X)
  r <- nrow(X)
  optimal_weigths <- optimal_weigths_sigma[1:(n-1)] # assumes no intercept as asked
  optimal_sigma <- optimal_weigths_sigma[n]
  sum_cov = 0
  
  while (num_simulations > 0) {
    
    set.seed(num_simulations)
    sample_data <- X[sample(1:r, num_samples), ]
    sum_cov <- sum_cov + cov(sample_data[, 1], sample_data[, -1] %*% optimal_weigths)
    num_simulations = num_simulations - 1
    
  }
  
  return(as.numeric(sum_cov/optimal_sigma^2))
  
}

MSE <- function(X, optimal_weights) { # based on definition here: https://en.wikipedia.org/wiki/Mean_squared_error
  
  n <- nrow(X)
  return(sum((X[, 1] - X[, -1] %*% optimal_weights)^2)/n) # built for no-intercept models as asked
  
}

# model 1: lambda = 1
model1_optimal_weights_sigma <- ridge_opt(train_data, lambda = 1)
model1_optimal_weights <- model1_optimal_weights_sigma[1:(n_col - 1)]
model1_optimal_sigma <- model1_optimal_weights_sigma[n_col]
model1_train_MSE <- MSE(train_data, model1_optimal_weights)
model1_test_MSE <- MSE(test_data, model1_optimal_weights)
model1_AIC <- -2*model_log_likelihood(model1_optimal_weights, model1_optimal_sigma, as.matrix(train_data)) + 
  2*degrees_of_freedom(as.matrix(train_data), num_samples = 1000, 
                       num_simulations = 500, model1_optimal_weights_sigma)

# model 2: lambda = 100
model2_optimal_weights_sigma <- ridge_opt(train_data, lambda = 100)
model2_optimal_weights <- model2_optimal_weights_sigma[1:(n_col - 1)]
model2_optimal_sigma <- model2_optimal_weights_sigma[n_col]
model2_train_MSE <- MSE(train_data, model2_optimal_weights)
model2_test_MSE <- MSE(test_data, model2_optimal_weights)
model2_AIC <- -2*model_log_likelihood(model2_optimal_weights, model2_optimal_sigma, as.matrix(train_data)) + 
  2*degrees_of_freedom(as.matrix(train_data), num_samples = 1000, 
                       num_simulations = 500, model2_optimal_weights_sigma)

# model 3: lambda = 1000
model3_optimal_weights_sigma <- ridge_opt(train_data, lambda = 1000)
model3_optimal_weights <- model3_optimal_weights_sigma[1:(n_col - 1)]
model3_optimal_sigma <- model3_optimal_weights_sigma[n_col]
model3_train_MSE <- MSE(train_data, model3_optimal_weights)
model3_test_MSE <- MSE(test_data, model3_optimal_weights)
model3_AIC <- -2*model_log_likelihood(model3_optimal_weights, model3_optimal_sigma, as.matrix(train_data)) + 
  2*degrees_of_freedom(as.matrix(train_data), num_samples = 1000, 
                       num_simulations = 500, model3_optimal_weights_sigma)

# plot performance
MSE_data <- data.frame(log_10_lambda = c(0, 0, 2, 2, 3, 3), Legend = rep(c("Training", "Test"), 3), 
                       value = c(model1_train_MSE, model1_test_MSE, model2_train_MSE, 
                                 model2_test_MSE, model3_train_MSE, model3_test_MSE))

ggplot(MSE_data) + geom_line(aes(log_10_lambda, value, colour = Legend)) + theme_bw() + 
  geom_vline(xintercept = 2, linetype = "dotted") + 
  xlab("Log (base-10) of Hyper-Parameter 'lambda'") + ylab("Mean Squared Error") + 
  ggtitle("Finding the Optimal Hyper-Parameter")

AIC_data <- data.frame(log_10_lambda = c(0, 2, 3), value = c(model1_AIC, model2_AIC, model3_AIC))

ggplot(AIC_data) + geom_line(aes(log_10_lambda, value)) + theme_bw() + 
  geom_vline(xintercept = 2, linetype = "dotted") + 
  xlab("Log (base-10) of Hyper-Parameter 'lambda'") + ylab("Model AIC") + 
  ggtitle("Finding the Optimal Hyper-Parameter")

### Question 3

# read in data
tecator <- read_csv("tecator.csv", col_types = cols())

# drop columns not to be used for prediction
tecator <- tecator[, -c(1, 103, 104)]

# split into training/test
all_indices <- 1:(nrow(tecator))
set.seed(12345)
train_indices <- sample(all_indices, ceiling(0.5*nrow(tecator)))
test_indices <- all_indices[!(all_indices %in% train_indices)]
train_data <- tecator[train_indices, ]
test_data <- tecator[test_indices, ]

## linear regression

# model

# $$\mathbf{y} \sim N(\boldsymbol{\beta}\mathbb{X}, \sigma^2\cdot \mathbb{I})$$

# fit
linreg <- lm(Fat~., train_data)
y_train_pred <- predict(linreg)
y_test_pred <- predict(linreg, test_data)
train_mse <- sum((train_data$Fat - y_train_pred)^2)/nrow(train_data)
test_mse <- sum((test_data$Fat - y_test_pred)^2)/nrow(test_data)

cat("Training MSE is:", train_mse)
cat("\nTest MSE is: ", test_mse)

## lasso regression

# objective function

## $$\sum\limits_{i=1}^N{(y_i - \boldsymbol{\beta}\mathbb{X}_i)^2} + \lambda\sum\limits_{j=1}^p{|\beta_i|}$$

# helper function: degrees of freedom
lasso_degrees_of_freedom <- function(lasso_model, y, X, num_simulations, num_samples) {
  
  X <- as.matrix(X)
  y <- unlist(y)
  y_pred <- as.numeric(predict(lasso, X))
  r <- nrow(X)
  sum_cov = 0
  
  while (num_simulations > 0) {
    
    set.seed(num_simulations)
    sample_indices <- sample(1:r, num_samples)
    sample_X <- X[sample_indices, ]
    sample_y <- y[sample_indices]
    sample_y_pred <- y_pred[sample_indices]
    sum_cov <- sum_cov + cov(sample_y, sample_y_pred)
    num_simulations = num_simulations - 1
    
  }
  
  sigma_hat_square <- sum((y - y_pred)^2)/num_samples
  
  return(sum_cov/sigma_hat_square)
  
}

# fit
#lambda_choices <- c(10^-3, 10^-2, 10^-1, 1, 10, 100, 1000)
#zero_coeff <- c()
#sum_abs_val_params <- c()
#lasso_deg_freedoms <- c()
#for (lam in lambda_choices){
  
  lasso <- glmnet(x = as.matrix(subset(train_data, select = -Fat)),
                  y = unlist(train_data[, "Fat"]), alpha = 1
                  #, lambda = lam
                  , family = "gaussian")
  #sum_abs_val_params <- c(sum_abs_val_params, sum(abs(lasso$beta)))
  #zero_coeff <- c(zero_coeff, sum(lasso$beta == 0))
  
  #lasso_deg_freedoms <- c(lasso_deg_freedoms, 
  #                        lasso_degrees_of_freedom(
  #                          lasso, tecator[, "Fat"], subset(tecator, select = -Fat), 
  #                          num_simulations = 50, num_samples = 100)) # using full dataset rather than train / test, because why not!
  
#}

# plot results
plot(lasso)
  
#lasso_results <- data.frame(log_10_lambda = rep(-3:3, 2), 
#                            Legend = c(rep("# Parameters with Coefficients = 0", 7), 
#                                       rep("Sum of absolute values of Coefficients", 7)),
#                            value = c(zero_coeff, sum_abs_val_params))

#ggplot(lasso_results) + geom_line(aes(log_10_lambda, value, colour = Legend)) + theme_bw() + 
#  scale_x_continuous(breaks = -3:3) + theme(legend.position="bottom") + 
#  xlab("Log (base-10) of LASSO Hyper-Parameter 'lambda'") + ylab("Parameter Attributes") + 
#  ggtitle("Relationship between LASSO Lambda and parameter attributes")

#lasso_deg_freedom <- data.frame(log_10_lambda = rep(-3:3, 2), value = lasso_deg_freedoms)

lasso_deg_freedom <- data.frame(lambda = lasso$lambda, value = lasso$df)

ggplot(lasso_deg_freedom) + geom_point(aes(lambda, value)) + theme_bw() +
  #scale_x_continuous(breaks = -3:3) + 
  xlab("Log (base-10) of LASSO Hyper-Parameter 'lambda'") + 
  ylab("Model Degrees of Freedom") + ggtitle("Relationship between LASSO Lambda and Model Degrees of Freedom")

## ridge regression

# fit
zero_coeff <- c()
sum_params_squared <- c()
for (lam in lambda_choices){
  
  ridge <- glmnet(x = as.matrix(subset(train_data, select = -Fat)),
                  y = unlist(train_data[, "Fat"]), alpha = 0, lambda = lam)
  sum_params_squared <- c(sum_params_squared, sum(ridge$beta^2))
  zero_coeff <- c(zero_coeff, sum(ridge$beta == 0))
  
}

# plot results
ridge_results <- data.frame(log_10_lambda = rep(-3:3, 2), 
                            Legend = c(rep("# Parameters with Coefficients = 0", 7), 
                                       rep("Sum of squared values of Coefficients", 7)),
                            value = c(zero_coeff, sum_params_squared))

ggplot(ridge_results) + geom_line(aes(log_10_lambda, value, colour = Legend)) + theme_bw() + 
  scale_x_continuous(breaks = -3:3) + theme(legend.position="bottom") + 
  xlab("Log (base-10) of Ridge Hyper-Parameter 'lambda'") + ylab("Parameter Attributes") + 
  ggtitle("Relationship between Ridge Lambda and parameter attributes")

## lasso with CV

# fit
cv_means <- c()
cv_min_lambda <- c()
for (i in 1:100) {
  
  cv_lasso <- cv.glmnet(x = as.matrix(subset(train_data, select = -Fat)), 
                        y = unlist(train_data[, "Fat"]), alpha = 1, 
                        lambda = lambda_choices, nfolds = 3)
  cv_means <- cbind(cv_means, cv_lasso$cvm)
  cv_min_lambda <- c(cv_min_lambda, cv_lasso$lambda.min)
}
plot(cv_lasso)
# plot
ggplot(data.frame(log_10_lambda = -3:3, cv_score = rev(rowMeans(cv_means)))) + 
  geom_line(aes(log_10_lambda, cv_score)) + scale_x_continuous(breaks = -3:3) +
  geom_vline(xintercept = -3, linetype = "dotted") + 
  xlab("Log (base-10) of LASSO Hyper-Parameter 'lambda'") + ylab("Mean squared error across CV folds") + 
  ggtitle("Relationship between LASSO Lambda and CV Scores")

opt_lambda <- cv_lasso$lambda.min
cat("Optimal Lambda is:", opt_lambda)

model_optimal_lambda <- glmnet(x = as.matrix(subset(train_data, select = -Fat)),
                               y = unlist(train_data[, "Fat"]), alpha = 1, lambda = opt_lambda)
num_non_zero_params_opt_lambda <- sum(model_optimal_lambda$beta != 0)
cat("Number of variables chosen in the model:", num_non_zero_params_opt_lambda)

# comparing LASSO regression model with optimal lambda and another with log (base-10) lambda = -2
model_log_lambda_minus_2 <- glmnet(x = as.matrix(subset(train_data, select = -Fat)),
                                   y = unlist(train_data[, "Fat"]), alpha = 1, lambda = 10^-2)
y_test <- unlist(test_data[, "Fat"])
y_test_pred_opt_lambda <- predict(model_optimal_lambda, as.matrix(subset(test_data, select = -Fat)))
y_test_log_lambda_minus_2 <- predict(model_log_lambda_minus_2, as.matrix(subset(test_data, select = -Fat)))

num_non_zero_params_log_lambda_minus_2 <- sum(model_log_lambda_minus_2$beta != 0)

opt_lambda_SSE <- sum((y_test - y_test_pred_opt_lambda)^2) 
log_lambda_minus_2_SSE <- sum((y_test - y_test_log_lambda_minus_2)^2)

opt_lambda_deg_freedom <- nrow(test_data) - num_non_zero_params_opt_lambda - 1
log_lambda_minus_2_deg_freedom <- nrow(test_data) - num_non_zero_params_log_lambda_minus_2 - 1

opt_lambda_MSE <- opt_lambda_SSE / opt_lambda_deg_freedom
log_lambda_minus_2_MSE <- log_lambda_minus_2_SSE / log_lambda_minus_2_deg_freedom
test_statistic <- opt_lambda_MSE / log_lambda_minus_2_MSE

# Null Hypothesis: opt_lambda_MSE = log_lambda_minus_2_MSE
# Alternative Hypothesis: opt_lambda_MSE < log_lambda_minus_2_MSE

p_value <- pf(q = test_statistic, 
              df1 = opt_lambda_deg_freedom, 
              df2 = log_lambda_minus_2_deg_freedom)

cat("The p-value for the test of null hypothesis that optimal lambda works similar to log(lambda) = -2",
    "vs. the alternative that the former is better is:", p_value)
cat("We, therefore, cannot conclude confidently that the optimal",
    "lambda works significantly better than log(lambda) = -2")

# plot predictions
ggplot(data.frame(y_test, y_test_pred_opt_lambda)) + 
  geom_point(aes(y_test, y_test_pred_opt_lambda)) + 
  xlab("Test Labels") + ylab("Test Predictions") +
  ggtitle("Goodness of fit - LASSO model with optimised lambda")

## generative model

# estimate sigma: use MLE now ;-)
y_train <- unlist(train_data[, "Fat"])
y_train_pred <- predict(model_optimal_lambda, as.matrix(subset(train_data, select = -Fat)))
MLE_sigma_estimate <- sqrt(sum((y_train - y_train_pred)^2)/nrow(train_data))

# generate labels from distribution N(y_train_pred, MLE_sigma_estimate*I)
set.seed(12345)
y_generated <- rnorm(length(y_test), y_test_pred_opt_lambda, MLE_sigma_estimate)

# plot generated labels vs. original labels
ggplot(data.frame(y_test, y_generated)) + 
  geom_point(aes(y_test, y_generated)) + 
  xlab("Test Labels") + ylab("Generated Labels") +
  ggtitle("Goodness of fit - Generative LASSO model with optimised lambda")
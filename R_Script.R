library(readr)
library(kknn)
library(ggplot2)
library(data.table)

### Question 1

# read in data
optdigits <- readr::read_csv("optdigits.csv", col_names = FALSE)

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

# contingency tables for train and test datasets
train_c_table <- table(train_data$X65, fitted(model_30_NN_train))
test_c_table <- table(test_data$X65, fitted(model_30_NN_test))

cat("Contingency table for Training Dataset is: \n")
print(train_c_table)
cat("Contingency table for Test Dataset is: \n")
print(test_c_table)

# misclassification errors for the training and test data
training_misclass <- round((1-sum(diag(train_c_table))/sum(train_c_table))*100, 2)
test_misclass <- round((1-sum(diag(test_c_table))/sum(test_c_table))*100, 2)

cat("Training mis-classification rate is: ", training_misclass, "%", sep = "")
cat("Test mis-classification rate is: ", test_misclass, "%", sep = "")

# cases with (a) 'ground truth' class = '8', and (b) highest or lowest modelled probability of belonging to class '8'
class_8_probs_for_true_8 <- model_30_NN_train$prob[train_data$X65 == 8, 9]
class_8_indices <- which(train_data$X65 == 8)
sorted_prob_indices <- order(class_8_probs_for_true_8)
top_1_prob_data <- unlist(train_data[class_8_indices[sorted_prob_indices[length(sorted_indices)]], 1:64])
top_2_prob_data <- unlist(train_data[class_8_indices[sorted_prob_indices[length(sorted_indices) - 1]], 1:64])
bott_1_prob_data <- unlist(train_data[class_8_indices[sorted_prob_indices[1]], 1:64])
bott_2_prob_data <- unlist(train_data[class_8_indices[sorted_prob_indices[2]], 1:64])
bott_3_prob_data <- unlist(train_data[class_8_indices[sorted_prob_indices[3]], 1:64])

# transform 1x64 into 8x8
top_1_prob_data <- t(sapply(1:8, function(x) top_1_prob_data[((x-1)*8+1):(x*8)]))
top_2_prob_data <- t(sapply(1:8, function(x) top_2_prob_data[((x-1)*8+1):(x*8)]))
bott_1_prob_data <- t(sapply(1:8, function(x) bott_1_prob_data[((x-1)*8+1):(x*8)]))
bott_2_prob_data <- t(sapply(1:8, function(x) bott_2_prob_data[((x-1)*8+1):(x*8)]))
bott_3_prob_data <- t(sapply(1:8, function(x) bott_3_prob_data[((x-1)*8+1):(x*8)]))

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
  cross_entropy_train <- c(cross_entropy_train, sum(one_hot_y_train * -log(train_model$prob + 10^-15))/num_train_examples)
  cross_entropy_val <- c(cross_entropy_val, sum(one_hot_y_val * -log(val_model$prob + 10^-15))/num_val_examples)
  
}

missclass <- melt(data.table(k = 1:30, Training = missclass_train, Validation = missclass_val), "k",
                  variable.name = "Legend")
cross_entropy <- melt(data.table(k = 1:30, Training = cross_entropy_train, Validation = cross_entropy_val),
                      "k", variable.name = "Legend")

# plot misclassification rates and cross-entropy by 'k' for training and validation datasets
ggplot(missclass) + geom_line(aes(k, value, colour = Legend)) + theme_bw() + 
  geom_vline(xintercept = 3, linetype = "dotted") + scale_x_continuous(breaks = 1:30) +
  xlab("Hyper-Parameter 'k'") + ylab("Mis-classification Rates") + 
  ggtitle("Finding the Optimal Hyper-Parameter")

ggplot(cross_entropy) + geom_line(aes(k, value, colour = Legend)) + theme_bw() + 
  geom_vline(xintercept = 6, linetype = "dotted") + scale_x_continuous(breaks = 1:30) +
  xlab("Hyper-Parameter 'k'") + ylab("Cross-Entropy") + 
  ggtitle("Finding the Optimal Hyper-Parameter")

### Question 2

# read in data
parkinsons <- read_csv("parkinsons.csv")

# drop columns not to be used for prediction
parkinsons <- parkinsons[, -c(1, 2, 3, 4, 6)]

# bayesian model

## $$\mathbf{y} | \mathbb{X}, \lambda \sim N(w_0 + \mathbf{w}\mathbb{X}, \sigma^2\cdot \mathbb{I})$$
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
  par <- rnorm(n) # initial parameter values (to seed optimisation); random
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
model1_AIC <- -2*model_log_likelihood(model1_optimal_weights, model1_optimal_sigma, as.matrix(parkinsons)) + 
  2*degrees_of_freedom(as.matrix(parkinsons), num_samples = 1000, 
                       num_simulations = 50, model1_optimal_weights_sigma) # using full dataset rather than train / test, because why not!

# model 2: lambda = 100
model2_optimal_weights_sigma <- ridge_opt(train_data, lambda = 100)
model2_optimal_weights <- model2_optimal_weights_sigma[1:(n_col - 1)]
model2_optimal_sigma <- model2_optimal_weights_sigma[n_col]
model2_train_MSE <- MSE(train_data, model2_optimal_weights)
model2_test_MSE <- MSE(test_data, model2_optimal_weights)
model2_AIC <- -2*model_log_likelihood(model2_optimal_weights, model2_optimal_sigma, as.matrix(parkinsons)) + 
  2*degrees_of_freedom(as.matrix(parkinsons), num_samples = 1000, 
                       num_simulations = 50, model2_optimal_weights_sigma) # using full dataset rather than train / test, because why not!

# model 3: lambda = 1000
model3_optimal_weights_sigma <- ridge_opt(train_data, lambda = 1000)
model3_optimal_weights <- model3_optimal_weights_sigma[1:(n_col - 1)]
model3_optimal_sigma <- model3_optimal_weights_sigma[n_col]
model3_train_MSE <- MSE(train_data, model3_optimal_weights)
model3_test_MSE <- MSE(test_data, model3_optimal_weights)
model3_AIC <- -2*model_log_likelihood(model3_optimal_weights, model3_optimal_sigma, as.matrix(parkinsons)) + 
  2*degrees_of_freedom(as.matrix(parkinsons), num_samples = 1000, 
                       num_simulations = 50, model3_optimal_weights_sigma) # using full dataset rather than train / test, because why not!

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
  geom_vline(xintercept = 3, linetype = "dotted") + 
  xlab("Log (base-10) of Hyper-Parameter 'lambda'") + ylab("Model AIC") + 
  ggtitle("Finding the Optimal Hyper-Parameter")

# By using function RidgeOpt, compute optimal w parameters for 𝜆𝜆=1,𝜆𝜆=100 and 𝜆𝜆=1000.
# Use the estimated parameters to predict the motor_UPDRS values for training and test data and
# report the training and test MSE values. Which penalty parameter is most appropriate among the selected ones?
# Why is MSE a more appropriate measure here than other empirical risk functions?

# Use functions from step 3 to compute AIC (Akaike Information Criterion) scores for the Ridge models
# with values 𝜆𝜆=1,𝜆𝜆=100 and 𝜆𝜆=1000 and their corresponding optimal parameters w and 𝜎𝜎 
# computed in step 4. What is the optimal model according to AIC criterion? What is the theoretical 
# advantage of this kind of model selection compared to the holdout model selection done in step 4?
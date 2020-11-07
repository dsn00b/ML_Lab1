library(readr)
library(kknn)
library(ggplot2)
set.seed(12345)

### Question 1

# read in data
optdigits <- readr::read_csv("optdigits.csv", col_names = FALSE)

# treat Y variable "X65" as factor
optdigits$X65 <- as.factor(optdigits$X65)

# create train/validation/test partitions
all_indices <- 1:(nrow(optdigits))
train_indices <- sample(all_indices, ceiling(0.5*nrow(optdigits)))
val_test_indices <- all_indices[!(all_indices %in% train_indices)]
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
one_hot_y_train <- sapply(as.numeric(train_data$X65), function(x) {c(rep(0, x - 1), 1, rep(0, num_classes - x))})
one_hot_y_val <- sapply(as.numeric(val_data$X65), function(x) {c(rep(0, x - 1), 1, rep(0, num_classes - x))})

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
  cross_entropy_train <- c(cross_entropy_train, sum(one_hot_y_train %*% log(train_model$prob + 10^-15))/num_train_examples)
  cross_entropy_val <- c(cross_entropy_val, sum(one_hot_y_val %*% log(val_model$prob + 10^-15))/num_val_examples)
  
}

# plot misclassification rates and cross-entropy by 'k' for training and validation datasets

library(readr)
library(kknn)
set.seed(12345)

### Question 1

# read in data
optdigits <- readr::read_csv("optdigits.csv", col_names = FALSE)

# treat Y variable "X65" as factor
optdigits$X65 <- as.factor(optdigits$X65)

# create train/validation/test partitions
n <- nrow(optdigits)
all_indices <- 1:n
train_indices <- sample(all_indices, ceiling(0.5*n))
val_test_indices <- all_indices[!(all_indices %in% train_indices)]
val_indices <- sample(val_test_indices, ceiling(0.5*length(val_test_indices))) # 25% of all_indices == 50% of val_test_indices
test_indices <- val_test_indices[!(val_test_indices %in% val_indices)]
train_data <- optdigits[train_indices, ]
val_data <- optdigits[val_indices, ]
test_data <- optdigits[test_indices, ]

# train k-NN model with k = 30 using kknn function as asked
model_30_NN_train <- kknn(formula = X65~., kernel = "rectangular", train = train_data, test= train_data, k = 30)
model_30_NN_test <- kknn(formula = X65~., kernel = "rectangular", train = train_data, test= test_data, k = 30)

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

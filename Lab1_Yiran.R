library(kknn)
library(glmnet)

#Assignment 1
#1.1
digits <- read.csv("optdigits.csv", header = F)
colnames(digits)[ncol(digits)] <- "number"
digits$number <- as.factor(digits$number)
n <- dim(digits)[1] 
set.seed(12345)
id <- sample(1:n, floor(n*0.5)) 
train <- digits[id,]
id1 <- setdiff(1:n, id) 
set.seed(12345) 
id2 <- sample(id1, floor(n*0.25)) 
validation <- digits[id2,]
id3 <- setdiff(id1,id2) 
test <- digits[id3,]

#1.2
knn_train <- kknn(number ~., train, train, k = 30, kernel = "rectangular")
knn_test <- kknn(number ~., train, test, k = 30, kernel = "rectangular")
#Confusion matrix
pred_train<- fitted(knn_train)
confusion_train <- table(train$number, pred_train)
pred_test <- fitted(knn_test)
confusion_test <- table(test$number, pred_test)
print(confusion_train)
print(confusion_test)

print(diag(confusion_train)/rowSums(confusion_train))
print(diag(confusion_test)/rowSums(confusion_test))

#Misclassification errors
missclass <- function(X,X1) { 
  n = length(X)
  return(1-sum(diag(table(X,X1)))/n)
}
mismatch_rate_train <- missclass(train$number, pred_train) #0.04500262
mismatch_rate_test <- missclass(test$number, pred_test) #0.05329154
cat("The mismatch rate for train data is:", mismatch_rate_train, 
    "\nThe mismatch rate for test data is:", mismatch_rate_test)

#Comment on the quality of predictions for different digits and on the overall prediction quality.

# For train data, the quality of predicting different digits is with a large range between 91% and 100%.
# The prediction for 0 is 100%, and following with 6, 2, 3 and 7 with 97% to 99% rate. While the lower ones are for digits 4, 9 and 1 with about 91%.
# This could indicate that even for train data itself, the model does not fit quite well.
# For test data, the accuracy rates for prediction on digits do not have a huge difference than for train data. Some of them are higher than the train data prediction rate such as for 1, 6, 7 and 9.
# The highest one is 100% for digit 6. And other ones are below the train ones with a lowest rate for digit 4 being 86%.
# As we can see here, the rates for digits comparing with them for train data do not have an obvious decreasing trending, which means the model generalizes well.
# 
# For the overall mismatch rate 0.045 for train data and 0.053 for test data, we can see that the rates for both of them are not low enough to be a relatively good predict model.
# Because the difference is also not very high, this can mean that the model is at least not over fitting.

#1.3
train_eight <- train[which(train$number==8),]
train_eight$prob <- knn_train$prob[which(train$number==8), "8"]
sortedResult <- sort(train_eight[, "prob"], index.return = T)
sortedResult$x[1:3] #0.1000000 0.1333333 0.1666667
hard_ids <- sortedResult$ix[1:3] #50  43 136
tail(sortedResult$x, n=c(2)) #1 1
easy_ids <- tail(sortedResult$ix, n=c(2)) #179 183
plotHeatmap <- function(index){
  heatmap(matrix(as.numeric(train_eight[index, 1:64]), 8, 8, byrow = T), Colv = NA, Rowv = NA)
}
for (i in c(hard_ids, easy_ids)) {
  plotHeatmap(i)
}
# The cases that has lowest accuracy rates are hard to recognized as digits 8 when visualzing them as well.
# While the 100% accurate cases are very easy to see as 8.

#1.4
e_t <- 0
e_v <- 0
for (k in 1:30) {
  kt <- kknn(number ~., train, train, k = k, kernel = "rectangular")
  e_t[k] <- missclass(train$number, fitted(kt))
  kv <- kknn(number ~., train, validation, k = k, kernel = "rectangular")
  e_v[k] <- missclass(validation$number, fitted(kv))
}
plot(1:30, e_v, ylim = c(0,0.06), type = "l", col = "red", xlab = "K", ylab = "miss-classification error rate")
lines(1:30, e_t, ylim = c(0,0.06), col = "blue")
legend(2, 0.06, legend=c("validation", "train"),
       col=c("red", "blue"), lty=1:1, cex=0.8)
#As k increases, the complexity of the model also increases. As we can see in the plot, the error rate for the validation data decreases first and then goes up quickly as K increases.
#While the error rate for the train data starts with 0 when k is 0, and then goes up gradually as K goes up.

#The optimal K would be 3 as at that point, the value for validation rate is the lowest and for train data is also quite low.

#From the perspective of bias-variance trade-off, low bias is not always perfect as it can lead to a higher variance(when error of train is 0, the error of validation is not the lowest).
#When bias is a little bit higher, the variance can be a bit lower as it can generalize new data better. But after certain point, when bias gets too high, the error rate for both train and validation data will be high. In this case, it would be not a suitable model.

k_test_optik <- kknn(number ~., train, test, k = 3, kernel = "rectangular")
cat("Test mis-classification rate when k=3 is:", missclass(test$number, fitted(k_test_optik)),
    "\nTrain mis-classification rate when k=3 is:", e_t[3],
    "\nValidation mis-classification rate when k=3 is:", e_v[3])
#The test error being 2.40% with K=3 is a little bit higher than validation data and about 1.25% higher than the train data. I would say the quality of the model with K=3 is acceptable.

#1.5
r_v <- 0
for (k in 1:30) {
  kv <- kknn(number ~., train, validation, k = k, kernel = "rectangular")
  s <- 0
  for (i in 0:9) {
    s <- s + sum(log(kv$prob[which(validation$number==i), toString(i)] + 1e-15)/nrow(kv$prob))
  }
  r_v[k] <- -s
}
plot(1:30, r_v, type = "l", xlab = "K", ylab = "cross entropy")
#The optimized K value is 6, because at that point, the risk value is the lowest with 0.1191288
#Cross entropy can detect difference between models that may have same result of misclassification rates.


#Assignment 2
#2.1
#to minimize: -logP(D|w,sigma) + lamda * ||w||^2
#y ~ N(w0 + w%*%X, sigma^2*I), w ~ N(0, sigma^2/lamda*I), where I = t(X)%*%X

#2.2
parkinson <- read.csv("parkinsons.csv")
#scale: (data-mu)/sd
parkinson_x <- parkinson[c(5,7:22)]
mu_features <- apply(parkinson_x, 2, mean) 
sd_features <- apply(parkinson_x, 2, sd)
parkinson_x <- as.data.frame(sapply(1:ncol(parkinson_x), function(x) (parkinson_x[x] - mu_features[x])/sd_features[x]))

n_all <- dim(parkinson_x)[1]
set.seed(12345)
id <- sample(1:n_all, floor(n_all*0.6)) 
train <- parkinson_x[id,]
id1 <- setdiff(1:n_all, id) 
test <- parkinson_x[id1,]

#2.3
#a
Loglikelihood <- function(param, Y, X){
  nf <- ncol(X)
  w <- param[1:nf]
  sigma <- param[nf+1]
  n <- nrow(X)
  result <- -(n/2) * log(2 * pi * sigma^2) - 1/(2 * sigma^2) * sum((Y - X %*% w)^2)
  return(result)
}
#b
Ridge <- function(param, lamda, Y, X){
  nf <- ncol(X)
  w <- param[1:nf]
  return(lamda * sum(w^2) - Loglikelihood(param, Y, X))
}
#c
RidgeOpt <- function(lamda, Y, X){
  nf <- ncol(X)
  return(optim(c(rnorm(nf), 1), Ridge, lamda = lamda, Y = Y, X = X, method = "BFGS"))
}
#d
DF <- function(param, lamda, X, Y, it_times=100){
  nf <- ncol(X)
  w <- param[1:nf]
  sigma <- param[nf+1]
  # sum_cv <- 0
  # for (i in 1:it_times) {
  #   sum_cv <- sum_cv + cov(Y, X %*% w)
  # }
  # return(sum_cv/(sigma^2))
  sum(diag(solve(t(X)%*%X + lamda * diag(dim(X)[2])) %*% (t(X) %*% X)))
}

#2.4
MSE <- function(Y, Yi) {
  n <- length(Y)
  sum((Y-Yi)^2)/n
}
#train
Ytr <- train$motor_UPDRS
Xtr <- as.matrix(train[-1])
result1 <- RidgeOpt(1, Ytr, Xtr)
MSE(Ytr,  Xtr %*% result1$par[1:ncol(Xtr)]) #0.8732769
result2 <- RidgeOpt(100, Ytr, Xtr)
MSE(Ytr,  Xtr %*% result2$par[1:ncol(Xtr)]) #0.8790672
result3 <- RidgeOpt(10000, Ytr, Xtr)
MSE(Ytr,  Xtr %*% result3$par[1:ncol(Xtr)]) #0.9678104
#test
Yt <- test$motor_UPDRS
Xt <- as.matrix(test[-1])
MSE(Yt,  Xt %*% result1$par[1:ncol(Xt)]) #0.9290358
MSE(Yt,  Xt %*% result2$par[1:ncol(Xt)]) #0.9262726
MSE(Yt,  Xt %*% result3$par[1:ncol(Xt)]) #0.9872191
#The least error is from when lambda equals to 100.
#Why is MSE a more appropriate measure here than other empirical risk functions?

#2.5
AICscore <- function(result, lamda, Y, X){
  -2*Loglikelihood(result$par, Y, X) + 2*DF(result$par, lamda, X, Y)
}
AICscore(result1, 1, Ytr, Xtr) #9553.596
AICscore(result2, 100, Ytr, Xtr) #9569.044
AICscore(result3, 10000, Ytr, Xtr) #9892.271
#What is the optimal model according to AIC criterion? 
#Lambda equals to 1000 is the optimal one.
#What is the theoretical advantage of this kind of model selection compared to the holdout model selection done in step 4?


#Assignment 3
tecator <- read.csv("tecator.csv")
n <- dim(tecator)[1]
set.seed(12345)
id <- sample(1:n, floor(n*0.5)) 
train <- tecator[id,]
id1 <- setdiff(1:n, id) 
test <- tecator[id1,]
#3.1
#If it is a linear regression, one of the underlying model could be y~wX.
R_Squared <- function(Y, Yi){
  1 - mean((Y-Yi)^2)/var(Y)
}
lr <- lm(Fat ~ .-Sample-Protein-Moisture, data = train)
summary(lr) #Multiple R-squared: 1,	Adjusted R-squared:  0.9994
#From the R-squared value, we can see that the model is fitting the train data very well. But could have overfitting problem.
R_Squared(test$Fat, predict(lr, test)) #-3.316342
#The model does not generalize well on test data. The model can be seen as overfitting because it is really bad fitting on the test data as the R square value is negative.

#3.2
#it should minimize: loglikelihood + lamda*|w| (this second term is the penalty term)

#3.3
lasso <- glmnet(as.matrix(train[,2:101]), train$Fat, alpha = 1, family = "gaussian")
plot(lasso, xvar = "lambda", label = T)
#From the plot, it shows that with lambda increases, all the coefficients get punished till they become to zero eventually.
#The coefficients with higher values, no matter is positive or negative, have higher influence on the Y value we want to predict here.
lasso$lambda[which(lasso$df == 3)] #0.8530452 0.7772630 0.7082131

#3.4
plot(lasso$lambda, lasso$df)
#As the lambda increases, the df drops. When lambda increases to 1, the df drops from over 30 to around 0 and stay there afterwards.
#This could mean that with a higher penalty, more and more features appear to be not very relevant to the predicted value.

#3.5
ridge <- glmnet(as.matrix(train[,2:101]), train$Fat, alpha = 0, family = "gaussian")
plot(ridge, xvar = "lambda", label = T)
#The coefficients drop gradually and smoothly with the ridge regression. And the decreasing rate between coefficients are closer with one another.
#The lambda values appear to be much bigger(thousands times) than the ones in lasso regression. 
#And all of the coefficients do not become to zero no matter how lambda changes.
plot(ridge$lambda, ridge$df)
#The df value does not change with lambda changes. It stays at 100 (the number of features) all the way till the end.
#This means in Ridge regression, the w values are not likely to become to zero as lasso regression. 
#Because the penalty term is more like a round area instead of a linear factor.

#3.6
lassocv <- cv.glmnet(as.matrix(train[,2:101]), train$Fat, type.measure = "mse", alpha = 1, family = "gaussian")
plot(x=log(lassocv$lambda), y=lassocv$cvm, xlab = "log(lambda)", ylab = "cv score")
#log(lassocv$lambda[51])  lassocv$cvm[51]
#The cv score stayed as very slow when lambda is low and increases when log(lambda) is around -2.5 all the way up 
#untill it gets around 125 with log(lambda) being as around 0. Then it goes slightly down and then goes up again.
#The optima lambda value is 0.05744535 (log value is -2.856921) with the lowest cv error of 13.50853. 
#8 variables are chosen.
#It is statistically slighly better than log(lambda)=-2.
test_predict <- predict(lassocv, newx = as.matrix(test[,2:101]), s = "lambda.min")
plot(x = test$Fat, y = test_predict)
#The model fits the test data pretty well since the scatter points are most likely linear with not so far off residuals.

#3.7
train_predict <- predict(lassocv,newx = as.matrix(train[,2:101]), s = "lambda.min")
set.seed(12345)
test_predict_new <- test$Fat + rnorm(nrow(test), sd = sd(train$Fat - train_predict))
plot(x = test$Fat, y = test_predict_new)
#The generation has higher variance and overall it follows the trend. So we can conclude that the model works fine.






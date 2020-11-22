#load library
library(readr)

#import dataset
data <- read.csv(file.choose())

# Exploratory Data Analysis
# drop feature | remove animal.name
data <- data[-1]

# table for diagnosis 
table(data$type)

# table or proportions with more informative labels
round(prop.table(table(data$type)) * 100, digits = 2)

# create normalization function | making data scale and unit free
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# normalize the data data 
#lapply = list apply = take one by one, converting to list and then normalize it
#as.data.frame > to convert from list to dataframe
data_n <- as.data.frame(lapply(data[1:16], normalize))

# confirm that normalization worked
summary(data_n)

# create training and testing data
library(caTools)

#randomizing the data
set.seed(0)
split <- sample.split(data, SplitRatio = 0.8)
data_train <- subset(data, split == TRUE)
data_test <- subset(data, split == FALSE)

# create labels for training and test data 
data_train_labels <- data_train$type
data_test_labels <- data_test$type

# Training a model on the data

# load the "class" library
#install.packages("class")
library(class)

#input train, input test, output
data_test_pred <- knn(train = data_train, test = data_test, cl = data_train_labels, k = 21)

#  Evaluating model performance
# actual values in rows and predicted values in column
confusion_test <- table(x = data_test_labels, y = data_test_pred)
confusion_test

Accuracy <- sum(diag(confusion_test))/sum(confusion_test)
Accuracy 

# Training Accuracy to compare against test accuracy | to get better accuracy
data_train_pred <- knn(train = data_train, test = data_train, cl = data_train_labels, k=18)

confusion_train <- table(x = data_train_labels, y = data_train_pred)
confusion_train

Accuracy_train <- sum(diag(confusion_train))/sum(confusion_train)
Accuracy_train

#Improving model performance
# Create the cross tabulation of predicted vs. actual
library("gmodels")
CrossTable(x = data_test_labels, y = data_test_pred, prop.chisq=FALSE)


#Automating k value incrementing to find best model
pred.train <- NULL
pred.val <- NULL
error_rate.train <- NULL
error_rate.val <- NULL
accu_rate.train <- NULL
accu_rate.val <- NULL
accu.diff <- NULL
error.diff <- NULL

for (i in 1:39) {
  pred.train <- knn(train = data_train, test = data_train, cl = data_train_labels, k = i)
  pred.val <- knn(train = data_train, test = data_test, cl = data_train_labels, k = i)
  error_rate.train[i] <- mean(pred.train!=data_train_labels)#predicted and actual
  error_rate.val[i] <- mean(pred.val != data_test_labels)
  accu_rate.train[i] <- mean(pred.train == data_train_labels)
  accu_rate.val[i] <- mean(pred.val == data_test_labels)  
  accu.diff[i] = accu_rate.train[i] - accu_rate.val[i]
  error.diff[i] = error_rate.val[i] - error_rate.train[i]
}

knn.error <- as.data.frame(cbind(k = 1:39, error.train = error_rate.train, error.val = error_rate.val, error.diff = error.diff))
knn.accu <- as.data.frame(cbind(k = 1:39, accu.train = accu_rate.train, accu.val = accu_rate.val, accu.diff = accu.diff))

#install.packages("ggplot2")
library(ggplot2)
#grammer of graphics | aes is for beautification
errorPlot = ggplot() + 
  geom_line(data = knn.error[, -c(3,4)], aes(x = k, y = error.train), color = "blue") +
  geom_line(data = knn.error[, -c(2,4)], aes(x = k, y = error.val), color = "red") +
  geom_line(data = knn.error[, -c(2,3)], aes(x = k, y = error.diff), color = "black") +
  xlab('knn') +
  ylab('ErrorRate')

accuPlot = ggplot() + 
  geom_line(data = knn.accu[,-c(3,4)], aes(x = k, y = accu.train), color = "blue") +
  geom_line(data = knn.accu[,-c(2,4)], aes(x = k, y = accu.val), color = "red") +
  geom_line(data = knn.accu[,-c(2,3)], aes(x = k, y = accu.diff), color = "black") +
  xlab('knn') +
  ylab('AccuracyRate')

# Plot for Accuracy
plot(knn.accu[, c(4)], type = "b", xlab = "K-Value", ylab = "DifferenceInAccu") 

# Plot for Error
plot(knn.error[, c(4)], type = "b", xlab = "K-Value", ylab = "DifferenceInError") 
#import packages
import pandas as pd
import numpy as np

#load dataset
zoo = pd.read_csv("Zoo.csv")

#EDA
# Excluding animal name column for zoo
zoo = zoo.iloc[:, 1:] 

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
zoo_n = norm_func(zoo.iloc[:, :16])
zoo_n.describe()

X = np.array(zoo_n.iloc[:,:]) # Predictors 
Y = np.array(zoo['type']) # Target | Type for glass

#train - test split 
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#import kNN package from sklearn
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))

#Confusion matrix
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 

# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 


# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])

# package to do visualizations 
import matplotlib.pyplot as plt 

# train accuracy plot 
# r = red, o = circle, - = line
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
# b = blue, o = circle, - = line
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
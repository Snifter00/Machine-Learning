# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3]) #[3] is the index of the categorical variable we wish to encode
X = onehotencoder.fit_transform(X).toarray()

#encoding the dependent variable ##this is not needed as the dependent variable (salary) is not categorical
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

#Avoiding the dummy variable trap
X = X[:, 1:] #this removes the first column from X by indexing from column [1]. To avoid redudant dependencies.
#this is done automatically in this library but its safer to hard code this.

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling  -  not required when using multiple linear regression as the library will tae care of it automatically

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)  #this 'fit' is where regressor learns the correlations of the test set

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm 
# note that this library does not assume a constant (i.e. intercept) so we need to add a column of ones
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis =1) # .astype(int) to force column of ints, this avoids type error  

## Now we start backward elimination!!
"""Create an optimal matrix of features"""
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # take matrix X of features, then at each step we remove a feature. Hence, we must specify all members of the matrix to start with.
#Create a new regressor from StatsModel library. 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Fits OLS to x and y. endog = dependent variable, exog = array of observations
regressor_OLS.summary()

#Remove least significant variable (i.e. the highest p value)
X_opt = X[:, [0, 1, 3, 4, 5]]  # take all the rows, and the selected columns.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #ordinary least squares
regressor_OLS.summary()

#Remove next least significant variable
X_opt = X[:, [0, 3, 4, 5]]  # take all the rows, and the selected columns.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #ordinary least squares
regressor_OLS.summary()

#Remove next least significant variable
X_opt = X[:, [0, 3, 5]]  # take all the rows, and the selected columns.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #ordinary least squares
regressor_OLS.summary()

#Remove next least significant variable
X_opt = X[:, [0, 5]]  # take all the rows, and the selected columns.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #ordinary least squares
regressor_OLS.summary()
##Stop eliminating as no variables with p > 0.05! R&D spend is a powerful predictor of profit

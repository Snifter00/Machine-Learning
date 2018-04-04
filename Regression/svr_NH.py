# Support Variable Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Splitting the dataset into the Training set and Test set
    # Only use if you have a large data set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
    # Most of our libraries don't require this - but this one does! SVR model is less common
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)  #scale the matrix of features 'X' by applying the fit_transform method
y = sc_y.fit_transform(y)  #scale the independent variable 'y'. Scaling counts as transforming

# Fitting the SVR Model to the dataset
# Create your regressor here
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')   #we choose kernel = rbf (gaussian) as we know linear is inappropriate
regressor.fit(X, y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]])))) 
# we need to transform the '6.5' so it is suited to the regressor. Transform expects an array.
# Two sets of [] to specify array (one set = vector!)
# NB it is already fitted, so only 'transform' method required.
#sc_Y.inverse_transform is called in order to convert salary back to unscaled, 'real' value.


# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


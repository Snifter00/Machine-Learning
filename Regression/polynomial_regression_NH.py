# Polynomial Regression

## We need to fit a linear regression to the dataset followed by a polynomial linear regression.
## So, two regressors

# Importing the libraries
import numpy as np #useful for maths
import matplotlib.pyplot as plt #makes nice plots
import pandas as pd  #this is used to import data sets

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# 'position' column is already encoded. So, don';t need to include in matrix of feaures
X = dataset.iloc[:, 1:2].values  #this makes sure X is a matrix (even though its a single colummn matrix!)
y = dataset.iloc[:, 2].values
#make sure x is a matrix (of features) and y is a vector. You can tell this from 'Size' in variable explorer

# No training or test sets required Two reasons:
# 1. The dataset is small, not enough data; 2. We require a v accurate prediction.
# NB: No feature scaling required. The Linear regression library does this for us.

# Fitting Linear regression to the dataset
##THink we only make this for comparison purposes!
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynomial regression to the dataset
    ### We achieve this by creating polynomial features and then ###
    ### fitting these to the linear regression model. ###
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)  #transform X (which has one independent variable) to a new matrix of several independent variables
X_poly = poly_reg.fit_transform(X) # First fit object to X and then tranform X to X_poly
## NB: x^0 = 1. So, no need to add ones for constant b0.
## OK, now include this fit in a new linear regression model
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y) # Now our polynomial model is created!

# Visualise the Linear Regression results￼
plt.scatter(X, y, color = "red")  #plot grade reference vs salary
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth or bluff:(linear regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1) #this gives us a vector. Added to increase reolution of analysis.
X_grid = X_grid.reshape((len(X_grid), 1)) #this converts the vector to a matrix
plt.scatter(X, y, color = "red")  #plot grade reference vs salary
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = "green")  #by stipulating the poly_reg variable instead of X_poly, we avoid complications if we want to update X or add new data to it. X_poly is initialised to an existing matrix of features, x.
plt.title("Truth or bluff:(Polynomial regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Preducting a new result with Linear Regression
lin_reg.predict(6.5)  #substitute matrix 'X' (or 'X-grid') for value we wish to predict 

# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))

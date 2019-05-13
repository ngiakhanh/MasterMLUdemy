import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, dataset.shape[1]-1].values

#Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)# -*- coding: utf-8 -*-

#Feature Scaling (don't need)

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
LinearRegressor1 = LinearRegression()
LinearRegressor1.fit(X, y)
y_pred = LinearRegressor1.predict(np.reshape([6.5], (-1, 1)))

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
PolyRegressor = PolynomialFeatures(degree = 4)
X_poly = PolyRegressor.fit_transform(X)
X_test = PolyRegressor.fit_transform(np.reshape([6.5], (-1, 1)))

LinearRegressor2 = LinearRegression()
LinearRegressor2.fit(X_poly, y)
y_pred2 = LinearRegressor2.predict(X_test)

#Visualizing the Linear Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, LinearRegressor1.predict(X_grid), color = 'blue')
plt.scatter(6.5, y_pred, color = 'green')
plt.plot(X_grid, LinearRegressor2.predict(PolyRegressor.fit_transform(X_grid)), color = 'black')
plt.scatter(6.5, y_pred2, color = 'orange')
plt.title('Salary vs Title')
plt.xlabel('Title')
plt.ylabel('Salary')
plt.show()
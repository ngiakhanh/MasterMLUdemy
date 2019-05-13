import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, dataset.shape[1]-1].values

#Fitting the Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#Predicting a new result
y_pred = regressor.predict(np.reshape([6.5], (-1, 1)))

#Visualizing the results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.scatter(6.5, y_pred, color = 'green')
plt.title('Salary vs Title')
plt.xlabel('Title')
plt.ylabel('Salary')
plt.show()
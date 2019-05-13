import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Don't need feature scaling

#Fitting Simple Linear Regression to the Traing set
from sklearn.linear_model import LinearRegression
LinearPredict = LinearRegression()
LinearPredict.fit(X_train, y_train)

#Predicting the test set results
y_pred = LinearPredict.predict(X_test)

#Visualizing the traing set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, LinearPredict.predict(X_train), color = 'blue')
plt.scatter(X_test, y_test, color = 'green')
plt.title('Salary vs Exp')
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.show()
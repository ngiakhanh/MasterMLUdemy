import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, dataset.shape[1]-1].values

#Categorize non-numerical data
X[:, 3] = LabelEncoder().fit_transform(X[:, 3])

#Encode data
X = OneHotEncoder(categorical_features = [3]).fit_transform(X).toarray()

#Avoiding the Dummy Variables trap (don't need to)
X = X[:, 1:]

#Feature Scaling (don't need)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Traing set
from sklearn.linear_model import LinearRegression
LinearPredict = LinearRegression()
LinearPredict.fit(X_train, y_train)

#Predicting the test set results
y_pred = LinearPredict.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1), dtype = int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print (regressor_OLS.summary())
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print (regressor_OLS.summary())
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print (regressor_OLS.summary())
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print (regressor_OLS.summary())
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print (regressor_OLS.summary())
#Visualizing the traing set results
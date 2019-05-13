import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.chdir('C:\\Users\\NHK81HC\\Desktop\\Learn Machine Learning\\Master Machine Learning A-Z (Udemy)\\Machine Learning A-Z\\Part 1 - Data Preprocessing')
print (os.getcwd())

#Extract data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3:].values

#Handle missing values
X[:, 1:] = SimpleImputer(missing_values = np.nan, strategy = 'mean').fit_transform(X[:, 1:])

#Categorize non-numerical data
X[:, 0] = LabelEncoder().fit_transform(X[:, 0])
y = LabelEncoder().fit_transform(y)

#Encode data
X = OneHotEncoder(categorical_features = [0]).fit_transform(X).toarray()

#Splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Feature scaling
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
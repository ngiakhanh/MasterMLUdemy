import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')
titles = list(dataset)

#Clean the text
import re
from nltk.stem.porter import PorterStemmer
stopwords = open("english.txt", "r").read().split('\n')
stopwords.pop()
corpus = []
for i in range(dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    
#Create the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting results
y_pred = classifier.predict(X_test)

#Make Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
Accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
Precision = cm[1,1]/(cm[1,1]+cm[0,1])
Recall = cm[1,1]/(cm[1,1]+cm[1,0])
F1_Score = 2 * Precision * Recall / (Precision + Recall)
print(Accuracy, Precision, Recall, F1_Score)
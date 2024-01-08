#import libraries used
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

#load the dataset
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
print (dataset)

stroke_dataset = dataset[dataset['stroke']==1]
heart_dataset = dataset[dataset['stroke']==1]

axes = stroke_dataset.plot(kind='scatter', x='bmi', y= 'avg_glucose_level', color ='blue', label='stroke' )
heart_dataset.plot(kind='scatter', x='bmi', y= 'avg_glucose_level', color ='red', label='stroke', ax=axes )

#Filter dataset to certain columns
features_dataset = dataset[['heart_disease','avg_glucose_level', 'bmi']]

X = np.asarray(features_dataset.values)

#Dependent variable

y = np.asarray(dataset['stroke'].values)

print(X)
print(y)

#Divide dataset into train and test -> Train (X, Y) -- Test (X,Y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#4088
print(X_train.shape)
#1022
print(X_test.shape)
#4088
print(y_train.shape)
#1022
print(y_test.shape)

# Replace NaN values with mean of the column
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

print(classification_report (y_test, y_predict))

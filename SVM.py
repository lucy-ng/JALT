#import libraries used
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#load the dataset
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
print (dataset)

stroke_dataset = dataset[dataset['stroke']==1]
heart_dataset = dataset[dataset['heart_disease']==1]

axes = stroke_dataset.plot(kind='scatter', x='bmi', y= 'avg_glucose_level', color ='blue', label='stroke' )
heart_dataset.plot(kind='scatter', x='bmi', y= 'avg_glucose_level', color ='red', label='stroke', ax=axes )

#Filter dataset to certain columns
features_dataset = dataset[['heart_disease','avg_glucose_level', 'bmi', 'stroke']]

X = np.asarray(features_dataset)

#Dependent variable

Y = np.asarray(dataset['stroke'])

print(X)
print(Y)

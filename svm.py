#import libraries used
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

#load the dataset
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
print (dataset)

stroke_dataset = dataset[dataset['stroke']==1]
heart_dataset = dataset[dataset['stroke']==0]
    
axes = stroke_dataset.plot(kind='scatter', x='bmi', y= 'avg_glucose_level', label='stroke', color = 'blue' )
heart_dataset.plot(kind='scatter', x='bmi', y= 'avg_glucose_level', label='no stroke', color = 'pink', ax=axes )

stroke1_dataset = dataset[dataset['stroke']==1]
heart1_dataset = dataset[dataset['stroke']==0]

axes = stroke1_dataset.plot(kind='kde', x='heart_disease', y= 'hypertension', label='stroke', color='pink' )
heart1_dataset.plot(kind='kde', x='heart_disease', y= 'hypertension', label='no stroke', ax=axes, color='blue' )

stroke2_dataset = dataset[dataset['stroke']==1]
heart2_dataset = dataset[dataset['stroke']==0]

axes = stroke2_dataset.plot(kind='density', x='hypertension', y= 'age', label='stroke', color = 'pink' )
heart2_dataset.plot(kind='density', x='hypertension', y= 'age', label='no stroke', color = 'blue', ax=axes )

#Filter dataset to certain columns
features_dataset = dataset[['bmi','avg_glucose_level', 'age', 'gender', 'hypertension', 'heart_disease', 'smoking_status' ]]

X = np.asarray(features_dataset.values)

#Dependent variable

y = np.asarray(dataset['stroke'].values)


#import libraries used
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#load the dataset
dataset = pd.read_csv("healthcare-dataset-stroke-data.csv")
dataset.tail()

stroke_dataset = dataset[dataset['stroke']==1]
stroke1_dataset = dataset[dataset['stroke']==0]

axes = stroke_dataset.plot(kind='scatter', x='bmi', y= 'avg_glucose_level', color ='blue', label='stroke' )
stroke1_dataset.plot(kind='scatter', x='bmi', y= 'avg_glucose_level', color ='red', label='stroke', ax=axes )

dataset.dtypes

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


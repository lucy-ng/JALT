# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:51:36 2023
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, mean_squared_error

# load dataset
data = pd.read_csv("healthcare-dataset-stroke-data.csv").dropna()
print(data)

# plot avg_glucose_level against bmi
avg_glucose_level = np.array(data["avg_glucose_level"]).reshape(4909, 1) # X
bmi = np.array(data["bmi"]).reshape(4909, 1) # y

plt.scatter(avg_glucose_level, bmi, s=0.5)
plt.title("Correlation between avg_glucose_level and bmi")
plt.xlabel("avg_glucose_label")
plt.ylabel("bmi")
plt.show()

# decision tree regressor model
# as values are continuous
decision_tree = DecisionTreeRegressor(criterion='squared_error')

# use K-Fold cross validation
kf = KFold(5, shuffle=True)
fold = 1

for train_index, validate_index in kf.split(avg_glucose_level, bmi):
    decision_tree.fit(avg_glucose_level[train_index], bmi[train_index])
    bmi_test = bmi[validate_index]
    bmi_pred = decision_tree.predict(avg_glucose_level[validate_index])
    print(f"Fold #{fold}, Training Size: {len(avg_glucose_level[train_index])}, Validation Size: {len(avg_glucose_level[validate_index])}")
    print('Accuracy: %.2f' % mean_squared_error(bmi_test, bmi_pred))
    fold += 1

# plot hypertension against age
hypertension = data["hypertension"]

# decision tree classifier model
decision_tree = DecisionTreeClassifier(criterion='entropy')

# split model and train data

# confusion matrix

# plot heart_disease against age
heart_disease = data["heart_disease"]

# decision tree classifier model
decision_tree = DecisionTreeClassifier(criterion='entropy')

# split model and train data

# confusion matrix
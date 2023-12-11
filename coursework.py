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
# contains 'NA' values so drop these values
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
glucose_bmi_decision_tree = DecisionTreeRegressor(criterion='squared_error')

# use K-Fold cross validation
kf = KFold(5, shuffle=True)
fold = 1

for train_index, validate_index in kf.split(avg_glucose_level, bmi):
    glucose_bmi_decision_tree.fit(avg_glucose_level[train_index], bmi[train_index])
    bmi_test = bmi[validate_index]
    bmi_pred = glucose_bmi_decision_tree.predict(avg_glucose_level[validate_index])
    print(f"Fold #{fold}, Training Size: {len(avg_glucose_level[train_index])}, Validation Size: {len(avg_glucose_level[validate_index])}")
    print('Root Mean Squared Error: %.2f' % np.sqrt(mean_squared_error(bmi_test, bmi_pred)))
    fold += 1

# plot hypertension against age
age = np.array(data["age"]).reshape(4909, 1) # X
hypertension = np.array(data["hypertension"]).reshape(4909, 1) # y

plt.scatter(age, hypertension, s=0.5)
plt.title("Correlation between age and hypertension")
plt.xlabel('age')
plt.ylabel('hypertension')
plt.show()

# decision tree classifier model
hypertension_age_decision_tree = DecisionTreeClassifier(criterion='entropy')

# split model and train data
age_train, age_test, hypertension_train, hypertension_test, = train_test_split(age, hypertension, test_size=0.25, random_state=42)
hypertension_age_decision_tree.fit(age_train, hypertension_train)

# accuracy score
hypertension_pred = hypertension_age_decision_tree.predict(age_test)
hypertension_age_accuracy = accuracy_score(hypertension_test, hypertension_pred)
print("Accuracy: %.2f" % hypertension_age_accuracy)

# confusion matrix
hypertension_age_cm = confusion_matrix(hypertension_test, hypertension_pred)
print(hypertension_age_cm)

# plot heart_disease against age
age = np.array(data["age"]).reshape(4909, 1) # X
heart_disease = np.array(data["heart_disease"]).reshape(4909, 1) # y

plt.scatter(age, heart_disease, s=0.5)
plt.title("Correlation between age and heart_disease")
plt.xlabel('age')
plt.ylabel('heart_disease')
plt.show()

# decision tree classifier model
heart_disease_age_decision_tree = DecisionTreeClassifier(criterion='entropy')

# split model and train data
age_train, age_test, heart_disease_train, heart_disease_test = train_test_split(age, heart_disease, test_size=0.25, random_state=42)
heart_disease_age_decision_tree.fit(age_train, heart_disease_train)

# accuracy score
heart_disease_pred = hypertension_age_decision_tree.predict(age_test)
heart_disease_age_accuracy = accuracy_score(heart_disease_test, heart_disease_pred)
print("Accuracy: %.2f" % heart_disease_age_accuracy)

# confusion matrix
heart_disease_age_cm = confusion_matrix(heart_disease_test, heart_disease_pred)
print(heart_disease_age_cm)
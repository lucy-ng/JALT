#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:14:50 2023

@author: t_o_s_h
"""
# =============================================================================
# Analysis 
#   Hear_disease coefficient is positive and around 0.07, this indicates that 
# there is a positive relationship between the presence of heart disease and
# stroke occurrences.
#   Hypertension coefficient is positive and around 0.04,
#this implies that there is a positive assiociation between hypertension and the
#probability of stroke occurrences. The coefficient value, although positive, 
# is somewhat smaller than that of heart_disease, indicating a lesser impact 
# but still a noticeable influence.
#   Age coefficient is positive but close to 0, this suggests thar there is 
# a minimal relationship between age and stroke. This model indicates that age 
# has the least influence among other factors on the likelihood of stroke occurrences. 
# Overall, accordign to linear regression model heart disease has a relatively 
# stronger relationship with stroke occurences. 
# =============================================================================


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('healthcare-dataset-stroke-data.csv')


features = ['age', 'hypertension', 'heart_disease']
X = data[features]
y = data['stroke']


model = LinearRegression()
model.fit(X, y)


coefficients = model.coef_

plt.figure(figsize=(5, 1))
plt.barh(features, coefficients)
plt.xlabel('Coefficient Value')
plt.title('Linear Regression')
plt.show()

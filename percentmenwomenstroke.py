#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:43:16 2023

@author: t_o_s_h
"""

#Analysis
# We can cleary see that looking at the data we have 
# we can see that female a most likely to have a stroke
#
#


import pandas as pd
import matplotlib.pyplot as plt

# load my data
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# filter data by those who had stroke 
stroke = data[data['stroke'] == 1]

# count the occurrences of stroke for each gender
gender = stroke['gender'].value_counts()

# Represent the result
gender.plot(kind='pie', autopct='%1.1f%%')
plt.title('Number of Men and Women who had a Stroke')
plt.xlabel('Gender')
plt.ylabel('')
plt.show()

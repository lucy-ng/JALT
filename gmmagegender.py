#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:54:39 2023

@author: t_o_s_h
"""

# AGE vs Gender for Stroke
# Count average age  for female and male who had stroke
# Average female age at having storoke is 67
# Averaege male age of having stroke is 68
#
#
#

#imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer


#load my data
data = pd.read_csv('healthcare-dataset-stroke-data.csv')


#filter data by stroke and gender
stroke = data[data['stroke'] == 1]
male = stroke[stroke['gender'] == 'Male']
female = stroke[stroke['gender'] == 'Female']

#take a list of 'age' 
features = ['age']
#taking age for female and male
malestroke = male[features]
femalestroke = female[features]


#handling missing values
imputer = SimpleImputer(strategy='mean')
male_stroke = imputer.fit_transform(malestroke)
female_stroke = imputer.fit_transform(femalestroke)

#initialize Gaussian Mixture model for male and female
male_stroke_mix = GaussianMixture(n_components=2)
female_stroke_mix = GaussianMixture(n_components=2)

#fit the model
male_stroke_mix.fit(male_stroke)
female_stroke_mix.fit(female_stroke)

#predicting clusters for male and female
malestrokecluster = male_stroke_mix.predict(male_stroke)
femalestrokecluster = female_stroke_mix.predict(female_stroke)

#creating plot
plt.figure(figsize=(10, 5))

#calculating percentage of male /female who had stroke
percentageOfmen = (len(male) / len(stroke)) * 100
percentageOfwomen = (len(female) / len(stroke)) * 100

#creating two plots, on the male and female
# Male
plt.subplot(1, 2, 1)
plt.scatter(male_stroke[:, 0], malestrokecluster, c=malestrokecluster, cmap='plasma')
plt.title('Male (Age vs Stroke )')
plt.xlabel(f'Age (Avg: {male_stroke.mean():.2f})')
plt.ylabel(f'Men ({percentageOfmen:.2f}%)')
plt.colorbar()

#Female
plt.subplot(1, 2, 2)
plt.scatter(female_stroke[:, 0], femalestrokecluster, c=femalestrokecluster, cmap='plasma')
plt.title('Female(Age vs Stroke) ')
plt.xlabel(f'Age (Avg: {female_stroke.mean():.2f})')
plt.ylabel(f'Women ({percentageOfwomen:.2f}%)')
plt.colorbar()

plt.tight_layout()
plt.show()


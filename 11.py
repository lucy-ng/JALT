#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:03:03 2023

@author: t_o_s_h
"""



# ANALYSIS
# The output shows frequency of stroke. With about 95.13% of instances showing no stroke
# ('0') and around 4.87% showing stroke cases ('1')
# This demostrates a significant difference between two, which may create challanges in accurate predictive modeling.
# It's importatn to consider that this impalance plays a crucial role in achieving accurate stroke predictions. 



import pandas as pd

# Load the dataset from data.csv
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Assuming 'stroke' is the column representing strokes (0: no stroke, 1: stroke)
frequencies_of_stroke = data['stroke'].value_counts(normalize=True)


# Convert frequencies to percentages
frequencies_of_stroke = frequencies_of_stroke * 100

print("Percentage of strokes in the dataset:")
print(frequencies_of_stroke)
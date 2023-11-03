#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:30:56 2023

@author: Antonina Pucko
"""

import pandas as pd


# load the dataset 
data = pd.read_csv("vgsales.csv")

# sort the dataset
data_sorted = data.sort_values(by='Global_Sales', ascending=False)

# top 10 games with the highest global sales
N = 10  
top_games = data_sorted.head(N)

print("Top", N, "most successful video game titles based on total global sales:")
print(top_games[['Rank', 'Name', 'Global_Sales']])


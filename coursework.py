# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:51:36 2023
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load dataset
data = pd.read_csv("healthcare-dataset-stroke-data.csv")
print(data)

# plot average glucose level against bmi
avg_glucose_level = data["avg_glucose_level"]
bmi = data["bmi"]

plt.scatter(avg_glucose_level, bmi, s=0.5)
plt.xlabel("avg_glucose_label")
plt.ylabel("bmi")
plt.show()

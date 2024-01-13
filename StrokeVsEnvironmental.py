# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:28:15 2024

@author: Jack Murphy
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# load data
data = pd.read_csv('healthcare-dataset-stroke-data.csv')


#           Pie Charts

# filter data by those who had stroke 
stroke = data[data['stroke'] == 1]

#function to draw the graphs    
def draw_graph(data, s1, s2, s3):
    holder = data[s1].value_counts()
    holder.plot(kind='pie', autopct='%1.1f%%')
    plt.title(s2)
    plt.xlabel(s3)
    plt.ylabel('')
    plt.show()

# stroke vs marriage situation
draw_graph(stroke, 'ever_married', '% of strokes by the Married', 'Married?')
# Overall marriage situation
draw_graph(data, 'ever_married', '% of Married people', 'Married?')

# stroke vs residence situation
draw_graph(stroke, 'Residence_type', '% of strokes by Urban people', 'Residence Type')
# Overall residence situation
draw_graph(data, 'Residence_type', '% of Urban people', 'Residence Type')

# stroke vs work situation
draw_graph(stroke, 'work_type', '% of strokes by work type', 'Work Type')
# Overall work situation
draw_graph(data, 'work_type', '% of people by work type', 'Work Type')

# Children muddy the waters by being an outlier, here are the graphs with children removed
# those who have never worked have also been removed, due to being so tiny a population and cluttering the data
# Used for reference for the below:  https://www.geeksforgeeks.org/how-to-drop-rows-that-contain-a-specific-string-in-pandas/?ref=ml_lbp
# temp stroke vs work excluding children
exclusionStrokeWork = stroke[stroke["work_type"].str.contains("children|Never_worked") == False]
draw_graph(exclusionStrokeWork, 'work_type', 'stroke% by workers', 'Work Type')
# temp population of work excluding children
exclusionWork = data[data["work_type"].str.contains("children|Never_worked") == False]
draw_graph(exclusionWork, 'work_type', '% by work type of those who have worked', 'Work Type')


#           Linear Regression

# Altering of data is necessary to use them for Linear Regression
# Convert the 3 environmental factors into integer booleans
# specifically, we're comparing the factors of: 
#           age (non environmental), being married, living in urban environments and being self-employed

#   for marriage
data['ever_married'] = data['ever_married'].replace(['Yes'], 1)
data['ever_married'] = data['ever_married'].replace(['No'], 0)
#   for residence
data['Residence_type'] = data['Residence_type'].replace(['Urban'], 1)
data['Residence_type'] = data['Residence_type'].replace(['Rural'], 0)
#   for work type
data['work_type'] = data['work_type'].replace(['Self-employed'], 1)
data['work_type'] = data['work_type'].replace(['Private'], 0)
data['work_type'] = data['work_type'].replace(['Govt_job'], 0)
data['work_type'] = data['work_type'].replace(['children'], 0)
data['work_type'] = data['work_type'].replace(['Never_worked'], 0)

# Now with updated data, filter only the 4 columns we're interested in (the 3 above plus 'age') into the X axis, and stroke on the y
features = ['age', 'ever_married', 'Residence_type', 'work_type']
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



#Analysis
# Marriage: the married population % of our data is 65.6%, but this spikes to 88.4% of the population with a stroke,
#       but this doesn't mention if they're still married, divorced, widowed, etc. There's a lot of information we haven't been given
#       Having been married also no doubt correlates heavily with age
#       Overall, I'd say that marriage correlates too strongly with things that cause strokes to be judged on its own
#       One must remove the other aspects first (mainly age)
# Residence: The Rural % of the overall population is 50.8%, but goes up to 54.2% of those with strokes
#       this suggests Urban people have had more strokes, this could be easier access to hospitals and thus it being recorded more.
#       However, all in all the countryside is generally more peaceful and probably correlates with aspects of life that decrease likelihood of strokes
# Work: Those who've never worked (excluding children) are only 0.4% of the population, but seem to get no strokes according to our data
#       Children, unsurprisingly go from 13.4% of the population, to 0.8% of the stroke-having population
#       The Self-employed go from 16% of the population to 26.1% of the stroke-having population.
#       They're in charge of their own destinies, possibly with no safety net, their having strokes (due to stress possibly) is unsurprising
#       Government workers have a slightly disproportionate number of strokes (going from 12.9% to 13.3%), 
#       However government jobs cover many different areas, from doctors to teachers to politicians and civil servants,
#       It would be hard to paint them all with the same brush, but some of those jobs definitely have long hours and are stress-inducing
#       Finally, the private sector doesn't change by much, and that's most people, which would be expected to be closer to the average

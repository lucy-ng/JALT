# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:28:15 2024

@author: Jack Murphy
"""

import pandas as pd
import matplotlib.pyplot as plt

# load my data
data = pd.read_csv('healthcare-dataset-stroke-data.csv')


# filter data by those who had stroke 
stroke = data[data['stroke'] == 1]

# stroke vs marriage situation
marriage = stroke['ever_married'].value_counts()
marriage.plot(kind='pie', autopct='%1.1f%%')
plt.title('Number of Married and Non-Married People who have had a stroke')
plt.xlabel('Married?')
plt.ylabel('')
plt.show()



# stroke vs residence situation
residence = stroke['Residence_type'].value_counts()
residence.plot(kind='pie', autopct='%1.1f%%')
plt.title('Number of Rural and Urban people who have had strokes')
plt.xlabel('Residence Type')
plt.ylabel('')
plt.show()



# stroke vs work situation
residence = stroke['work_type'].value_counts()
residence.plot(kind='pie', autopct='%1.1f%%')
plt.title('Number of Privately, Publically and Self-Employed people who have had strokes')
plt.xlabel('Work Type')
plt.ylabel('')
plt.show()


#Analysis
# Clearly those who have been married are more likely to get strokes, 
#       but this doesn't mention if they're still married, divorced, widowed, etc. There's a lot of information we haven't been given
#       Having been married also no doubt correlates heavily with age
# The data suggests Urban people have had more strokes, this could be easier access to hospitals and thus it being recorded more.
#       However, all in all the countryside is generally more peaceful and probably correlates with aspects of life that decrease likelihood of strokes
# As to Work Type, those in the private sector clearly have more strokes, however this could just be sheer numbers of privately employed people.
#       As expected, children are disproportionately unlikely to get strokes
#       We also can say nothing of the unemployed, we don't have the data

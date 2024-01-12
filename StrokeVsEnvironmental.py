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

#function to draw the graphs
def draw_graph(stroke, s1, s2, s3):
    if(stroke):
        holder = stroke[s1].value_counts()
    else:
        holder = data[s1].value_counts()
    holder.plot(kind='pie', autopct='%1.1f%%')
    plt.title(s2)
    plt.xlabel(s3)
    plt.ylabel('')
    plt.show()

# stroke vs marriage situation
draw_graph(False, 'ever_married', '% of strokes by the Married', 'Married?')
# Overall marriage situation
draw_graph(False, 'ever_married', '% of Married people', 'Married?')




# stroke vs residence situation
draw_graph(True, 'Residence_type', '% of strokes by Urban people', 'Residence Type')
# Overall residence situation
draw_graph(False, 'Residence_type', '% of Urban people', 'Residence Type')



# figure out how to keep colours consistent for both graphs

# stroke vs work situation
draw_graph(True, 'work_type', '% of strokes by work type', 'Work Type')
# Overall work situation
draw_graph(False, 'work_type', '% of people by work type', 'Work Type')




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

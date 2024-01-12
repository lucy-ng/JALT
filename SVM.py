#import libraries used
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

#load the dataset
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
print (dataset)

stroke_dataset = dataset[dataset['stroke']==1]
heart_dataset = dataset[dataset['stroke']==0]
    
axes = stroke_dataset.plot(kind='scatter', x='bmi', y= 'avg_glucose_level', label='stroke', color = 'blue' )
heart_dataset.plot(kind='scatter', x='bmi', y= 'avg_glucose_level', label='no stroke', color = 'pink', ax=axes )

stroke1_dataset = dataset[dataset['stroke']==1]
heart1_dataset = dataset[dataset['stroke']==0]

axes = stroke1_dataset.plot(kind='kde', x='heart_disease', y= 'hypertension', label='stroke', color='pink' )
heart1_dataset.plot(kind='kde', x='heart_disease', y= 'hypertension', label='no stroke', ax=axes, color='blue' )

stroke2_dataset = dataset[dataset['stroke']==1]
heart2_dataset = dataset[dataset['stroke']==0]

axes = stroke2_dataset.plot(kind='density', x='hypertension', y= 'age', label='stroke', color = 'pink' )
heart2_dataset.plot(kind='density', x='hypertension', y= 'age', label='no stroke', color = 'blue', ax=axes )


#Filter dataset to certain columns
features_dataset = dataset[['bmi','avg_glucose_level', 'age', 'gender', 'hypertension', 'heart_disease', 'smoking_status' ]]

X = np.asarray(features_dataset.values)

#Dependent variable

y = np.asarray(dataset['stroke'].values)

print(X)
print(y)

#Divide dataset into train and test -> Train (X, Y) -- Test (X,Y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#4088
print(X_train.shape)
#1022
print(X_test.shape)
#4088
print(y_train.shape)
#1022
print(y_test.shape)

# Replace NaN values with mean of the column
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

print(classification_report (y_test, y_predict, zero_division=0))

cm = confusion_matrix(y_test, y_predict)
print(cm)
accuracy_score(y_test,y_predict)

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = [ListedColormap(('red', 'green'))(i)], label = j)
plt.title('Stroke')
plt.xlabel('BMI')
plt.ylabel('Average Glucose Level')
plt.legend()
plt.show()
plt.show()

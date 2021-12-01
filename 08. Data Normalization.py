import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

df = pd.read_csv('D:/Projects/Machine Learning with Python/other_files/wisconsin_breast_cancer_dataset.csv')

print(df.describe().T)

df = df.drop('Unnamed: 32', axis = 1)
print(df.isnull().sum())

df = df.rename(columns = {'diagnosis' : 'Label'})
print(df.dtypes)

df['Label'].value_counts()

#Replace categorical values with numbers
categories = {'B': 0, 'M' : 1}
df['Label'] = df['Label'].replace(categories)

# Define dependent and independent variables
Y = df['Label'].values
X = df.drop(columns = ['Label', 'id'], axis = 1)


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state =10)

from sklearn import svm

model = svm.LinearSVC(max_iter=10000)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

from sklearn import metrics
print('Accuracy = ', metrics.accuracy_score(y_test, prediction))

#confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, prediction)
print(cm)

print("With Lung disease = ", cm[0,0] / (cm[0,0]+cm[1,0]))
print("No disease = ",   cm[1,1] / (cm[0,1]+cm[1,1]))





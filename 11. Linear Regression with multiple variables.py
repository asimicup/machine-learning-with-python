"""
#Heart disease
The effect that the independent variables biking and smoking 
have on the dependent variable heart disease 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('D:/Projects/Machine Learning with Python/other_files/heart.data.csv')
print(data.head())

data = data.drop('Unnamed: 0', axis = 1)

sns.lmplot(x = 'biking', y = 'heart.disease', data = data)
sns.lmplot(x = 'smoking', y = 'heart.disease', data = data)

x_df = data.drop('heart.disease', axis = 1)
y_df = data['heart.disease']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state = 42)

from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X_train, y_train) #Indep variables, dep. variable to be predicted
print(model.score(X_train, y_train)) #Prints the R^2 value, a measure of how well

prediction = model.predict(X_test)
print(y_test, prediction)

#Model is ready. Let us check the coefficients, stored as reg.coef_.
#These are a, b, and c from our equation. 
#Intercept is stored as reg.intercept_
print(model.coef_, model.intercept_)
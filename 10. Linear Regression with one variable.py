import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("D:/Projects/Machine Learning with Python/other_files/cells.csv")
print(data.head())

sns.lineplot(x = 'time', y = 'cells', data = data)

x = data.drop('cells', axis = 1)
print(x.dtypes)

y = data['cells']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#random_state can be any integer and it is used as a seed to randomly split dataset.
#By doing this we work with same test dataset evry time, if this is important.
#random_state=None splits dataset randomly every time

from sklearn import linear_model

model = linear_model.LinearRegression() #creates an instance of the model
model.fit(X_train, y_train) #Train the model or fits a linear model
prediction = model.predict(X_test)

print(model.score(X_train, y_train))#Prints the R^2 value, a measure of how well
#observed values are replicated by the model.

print(y_test, prediction)
print("Mean sqr error between y_test and prediction is: ", np.mean(prediction - y_test))

plt.scatter(prediction, prediction - y_test)
plt.hlines(y = 0, xmin = 200, xmax = 300)
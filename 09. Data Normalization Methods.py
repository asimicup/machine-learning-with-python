import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

df = pd.DataFrame(data= np.c_[housing['data'], housing['target']],
                     columns= housing['feature_names'] + ['target'])



print(df.describe().T)

# defining dependent and independent variables

Y = df['target'].values
X = df.drop(columns = ['target'], axis = 1)

sns.distplot(df['MedInc'], kde = False)
sns.distplot(df['Population'], kde = False) # outliers, 35682 is max but mean is 1425
sns.distplot(df['AveOccup'], kde = False) # large outliers, 1243 occupants?

X = X[['MedInc', 'AveOccup']].copy()
column_names = X.columns

sns.jointplot(x='MedInc', y='AveOccup', data = X, xlim =[0,10], ylim = [0,5])

################## Normalization Techniques ###################################

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

# Standard Scaler
#removes the mean and scales the data to unit variance.
# But, outliers have influence when computing mean and std. dev.

scaler1 = StandardScaler()
scaler1.fit(X)
X1 = scaler1.transform(X)
df1 = pd.DataFrame(data = X1, columns = column_names)
print(df1.describe())
sns.jointplot(x ='MedInc', y = 'AveOccup', data = df1)

#2 MinMaxScaler
#rescales the data set such that all feature values are in the range [0, 1] 
#For large outliers, it compresses lower values to too small numbers.
#Sensitive to outliers.
scaler2 = MinMaxScaler()
scaler2.fit(X)
X2 = scaler2.transform(X)
df2 = pd.DataFrame(data=X2, columns=column_names)
print(df2.describe())
sns.jointplot(x='MedInc', y='AveOccup', data=df2, xlim=[0,1], ylim=[0,0.005])  #Data scaled but outliers still exist

#3 RobustScaler
# the centering and scaling statistics of this scaler are based on percentiles 
#and are therefore not influenced by a few number of very large marginal outliers.
scaler3 = RobustScaler()
scaler3.fit(X)
X3 = scaler3.transform(X)
df3 = pd.DataFrame(data=X3, columns=column_names)
print(df3.describe())
sns.jointplot(x='MedInc', y='AveOccup', data=df3, xlim=[-2,3], ylim = [-2,3]) #Range -2 to 3


#4 PowerTransformer
# applies a power transformation to each feature to make the data more Gaussian-like
scaler4 = PowerTransformer()
scaler4.fit(X)
X4 = scaler4.transform(X)
df4 = pd.DataFrame(data=X4, columns=column_names)
print(df4.describe())
sns.jointplot(x='MedInc', y='AveOccup', data=df4) #

#5 QuantileTransformer
# has an additional output_distribution parameter allowing to match a 
# Gaussian distribution instead of a uniform distribution.
scaler5 = QuantileTransformer()
scaler5.fit(X)
X5 = scaler5.transform(X)
df5 = pd.DataFrame(data=X5, columns=column_names)
print(df5.describe())
sns.jointplot(x='MedInc', y='AveOccup', data=df5) #


#Dealing with null data

import pandas as pd
df = pd.read_csv('D:/Projects/Machine Learning with Python/other_files/manual_vs_auto.csv')

print(df.isnull())#Shows whether a cell is null or not
#Drop the entire column, if it makes sense
df = df.drop("Manual2", axis=1)
print(df.isnull().sum())   #Shows number of nulls in each column.

#If we only have handful of rows of null we can afford to drop these rows.
df2 = df.dropna()  #Drops all rows with at least one null value. 
#We can overwrite original df by equating it to df instead of df2.
#Or adding inplace=True inside
print(df2.head(25))  #See if null rows are gone.e.g. row 12

#If we have a lot of missing data then removing rows or columns
#may not be preferable.
#In such cases we can use Imputation technique.

print(df['Manual'].describe())  #Mean value of this column is 100.

df['Manual'].fillna(100, inplace=True)
print(df.head(25))   #Notice last entry in MinIntensity filled with 159.8


#another way to fill NaN is by filling with average of all auto columns from same row
import pandas as pd
import numpy as np

df = pd.read_csv('D:/Projects/Machine Learning with Python/other_files/manual_vs_auto.csv')

df['Manual'] = df.apply(lambda row: (round((row['Auto_th_2']+row['Auto_th_3']+row['Auto_th_3'])/3))
                        if np.isnan(row['Manual'])
                        else row['Manual'], axis=1)
print(df.head(25))

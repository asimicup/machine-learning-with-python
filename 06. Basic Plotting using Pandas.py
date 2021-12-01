
#Basic plotting of data from pandas dataframe.
#################################


import pandas as pd
df = pd.read_csv('D:/Projects/Machine Learning with Python/other_files/manual_vs_auto.csv')

#Pandas works with Matplotlib in the background

#To plot single histogram based on single value
#df['Manual'].plot(kind='hist', title='Manual Count')
df['Manual'].plot(kind='hist', title='Manual Count', bins=30, figsize=(12,10))

#To work only with Set 1 data we can create a new dataframe for that specific set
#and work with that dataframe. 

import pandas as pd
df = pd.read_csv('D:/Projects/Machine Learning with Python/other_files/manual_vs_auto.csv')
#Change Unnamed: 0 name to Image_set
df = df.rename(columns = {'Unnamed: 0':'Image_set'})
print(df.columns) 
set1_df = df[df['Image_set'] == 'Set1']
set1_df['Manual'].plot()

#Let's go back to all sets now.
import pandas as pd
df = pd.read_csv('D:/Projects/Machine Learning with Python/other_files/manual_vs_auto.csv')
#Change Unnamed: 0 name to Image_set
df = df.rename(columns = {'Unnamed: 0':'Image_set'})
#df['Manual'].plot()
# Sometimes we need to smooth data for better visualization.
#One way to Smooth is by averaging few points 
df['Manual'].rolling(3).mean().plot()
#Can do rolling mean or sum or anything else that makes sense.


#We can also graphically represent the statistics. 

import pandas as pd
df = pd.read_csv('D:/Projects/Machine Learning with Python/other_files/manual_vs_auto.csv')
#Change Unnamed: 0 name to Image_set
df = df.rename(columns = {'Unnamed: 0':'Image_set'})

print(df['Manual'].describe())

#In order to plot the relationship between Columns, we typically generate scatter plots

df.plot(kind='scatter', x='Manual', y='Auto_th_2', title='Manual vs Auto2')


#define all cell counts below 100 as low
#and above as high. Then let's plot using the new categories we defined.


import pandas as pd
df = pd.read_csv('D:/Projects/Machine Learning with Python/other_files/manual_vs_auto.csv')
#Change Unnamed: 0 name to Image_set
df = df.rename(columns = {'Unnamed: 0':'Image_set'})

#function to categorize low and high counts.
def cell_count(x):
    if x <= 100.0:
        return "low"
    else:
        return "high"

#Start by defining a new column title cell_count_index
#apply the function to categorize counts into low and high.
df["cell_count_index"] = df["Manual"].apply(cell_count)
print(df.head())
#Creates a new column called grain_category
#Can save to new csv
df.to_csv('D:/Projects/Machine Learning with Python/other_files/manual_vs_auto2.csv')

print(df.loc[20:30, ['Manual', 'cell_count_index']])

#we can plot by combining this cell count index information
df.boxplot(column='Manual', by='cell_count_index')





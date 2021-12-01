

# LOADING, VIEWING AND  UNDERSTANGING DATA

 
import pandas as pd

df = pd.read_csv('D:/Projects/Machine Learning with Python/other_files/cell_measurements.csv')
print(df.head())
print(df.columns)
df['Area'].plot(kind='hist', title='Area', bins=50)



import pandas as pd
df = pd.read_csv('D:/Projects/Machine Learning with Python/other_files/manual_vs_auto.csv')

print(df.info())  #Prvides an overview of the dataframe. 
print(df.shape)  #How many rows and columns

print(df)  #Shows a lot of stuff but truncated
print(df.head(7))  #Default prints 5 rows from the top
 
print(df.tail())   #Default prints 5 rows from the bottom

#First line in csv is considered header,
# so it prints it out every time
#First column is the index and it goes from 0, 1, 2, ....
#Index is not part of the data frame
#INdex is the unique identifier of a row, in our case a specific grain in a specific image
#Any of the other columns can be assigned as index if we know it is a unique identifier. 

import pandas as pd
df = pd.read_csv('D:/Projects/Machine Learning with Python/other_files/manual_vs_auto.csv')
print(df.index)  #Defines start and stop with step size. Not very exciting with default index
#But can be useful if we assign other column as index. 
df = df.set_index('Image')
print(df.head())
#View all column names.
print(df.columns)   #Image name column disappeared as it is used as index. 

#TO look at all unique entires. In this case, our 3 file names. 
print(df['Unnamed: 0'].unique())  

#If unnamed is bothering you then you can change the name.
df = df.rename(columns = {'Unnamed: 0':'Image_set'})
print(df.columns) 
#Missing data is encoded as NaN so we can work with them in a friendly manner. 
#Let us look at Manual column to see what it has.
print(df["Manual"])  #Shows NAN. We can fill it with something or ignore it or remove the column
 
 
#Pandas automatically recognizes correct data types.

print(df.dtypes)  

"""

#Similarly multiple column names can be changed at once. 
df = df.rename(columns = {'equivalent_diameter':'Diameter(um)', 
                          'Area':'Area(sq. um)',
                          'orientation':'orientation (deg)',
                          'MajorAxisLength':'Length (um)',
                          'MinorAxisLength':'Width (um)',
                          'Perimeter':'Perimeter (um)'})
print(df.dtypes)
"""

print(df.describe())  #Gives statistical summary of each column. 

#######################################################################


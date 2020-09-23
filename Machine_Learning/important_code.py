# ALl the Codes For the Data Science and Machine Learning which are state of the art and required


# ############# All the libraries we are going to use#############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# reading a csv file

data=pd.read.csv("file_name.csv")

# reading a file without the header

data = pd.read_csv("file_name.csv", header=None)


# Explicitly add the column names using the names parameter remember the parameter

data = pd.read_csv("file_name.csv", header=None, names=['x','y','z'])

# delimiters can be a different so you have to see which delimiters are used in the data

data = pd.read_csv("file_name.csv", sep=';')


# by passing the given headers with our own headers with the same code

data = pd.read_csv("file_name.csv", names=["a", "b", "c"])

# if you want to skip some initial rows you can do this by using the skiprows parameter
# and also the skipfooters to avoid some footer lines

data = pd.read_csv("file_name.csv", skiprows=2, skipfooter=2)

# you can read only first n rows if you want using the nrows parameter

data = pd.read_csv("file_name.csv", nrows=2)

# If you want to skip the blank lines you can also do this by using the skip_blank_lines as a parameter

data = pd.read_csv("file_names.csv", skip_blank_lines=False)
# means it will read the blank lines as well

# how to read the excel file with same name and the different sheet numbers so donot worry it is all good
df11_1 = pd.read_excel("Housing_data.xlsx", sheet_name='Data_Tab_1')
df11_2 = pd.read_excel("Housing_data.xlsx", sheet_name='Data_Tab_2')
df11_3 = pd.read_excel("Housing_data.xlsx", sheet_name='Data_Tab_3')


# Reading a general table using the read_table as a function
df13 = pd.read_table("Table_EX_1.txt")
df13

# if some separator between the data is given we can separate the data
# we can read the json tables,pdf tables and also the html table

# Numpy arrays behave like true numerical vectors and not the normal list values that are usually there

lst1 = [1, 2, 4]
array_numpy = np.array(lst1)
print(array_numpy)


# checking the type of anything we can do with the using type function

print(type(lst1))
print(array_numpy)


# Adding two list using the + operator results in the concat in list but the corresponding addition in arrays
lst2=[10,11,12]
array2 = np.array(lst2)
print(f"Adding two lists {lst1} and {lst2} together: {lst1+lst2}")

print(f"Adding two lists {lst1} and {lst2} together: {lst1+lst2}")



# Generating the arrays easily

print("A series of zeroes:",np.zeros(7))
print("A series of ones:",np.ones(9))
print("A series of numbers:",np.arange(5,16))
print("Numbers spaced apart by 2:",np.arange(0,11,2))
print("Numbers spaced apart by float:",np.arange(0,11,2.5))
print("Every 5th number from 30 in reverse order: ",np.arange(30,-1,-5))
print("11 linearly spaced numbers between 1 and 5: ",np.linspace(1,5,11))


# Printing with the sep as a parameter in the python

print("Dimension of this matrix: ", mat.ndim,sep='')
print("Size of this matrix: ", mat.size,sep='')
print("Shape of this matrix: ", mat.shape,sep='')
print("Data type of this matrix: ", mat.dtype,sep='')


# using ravel() to flat the array
b_flat = b.ravel()
print(b_flat)


#  using the random package
a = np.random.randint(1,100,30)
b = a.reshape(2,3,5)
c = a.reshape(6,5)
print ("Shape of a:", a.shape)
print ("Shape of b:", b.shape)
print ("Shape of c:", c.shape)


# using the index and slicing we got these values

arr = np.arange(0,11)
print("Array:",arr)
print("Element at 7th index is:", arr[7])
print("Elements from 3rd to 5th index are:", arr[3:6])
print("Elements up to 4th index are:", arr[:4])
print("Elements from last backwards are:", arr[-1::-1])
print("3 Elements from last backwards are:", arr[-1:-6:-2])

arr2 = np.arange(0, 21, 2)
print("New array:", arr2)
print("Elements at 2nd, 4th, and 9th index are:", arr2[[2, 4, 9]]) # Pass a list as a index to subset


#   we can use the  concept of conditional subsetting  and writing  the condition inside the brackets

mat = np.random.randint(10, 100, 15).reshape(3, 5)
print("Matrix of random 2-digit numbers\n", mat)
print("\nElements greater than 50\n", mat[mat > 50])

#   Sometimes we want the array of boolean values by just putting the condition
mat>50


# Matrix that values that ful fill the condition given
mat*(mat>50)

#  again using the random.randint

mat1 = np.random.randint(1, 10, 9).reshape(3,3)
mat2 = np.random.randint(1, 10, 9).reshape(3,3)
print("\n1st Matrix of random single-digit numbers\n",mat1)
print("\n2nd Matrix of random single-digit numbers\n",mat2)

print("\nAddition\n", mat1+mat2)
print("\nMultiplication\n", mat1*mat2)
print("\nDivision\n", mat1/mat2)
print("\nLineaer combination: 3*A - 2*B\n", 3*mat1-2*mat2)

print("\nAddition of a scalar (100)\n", 100+mat1)

print("\nExponentiation, matrix cubed here\n", mat1**3)
print("\nExponentiation, sq-root using pow function\n", pow(mat1, 0.5))


# Creating the Data Frame in Python
matrix_data = np.random.randint(1,20,size=20).reshape(5,4)
row_labels = ['A','B','C','D','E']
column_headings = ['W','X','Y','Z']

df = pd.DataFrame(data=matrix_data, index=row_labels, columns=column_headings)
print("\nThe data frame looks like\n",'-'*45, sep='')
print(df)


#   Some standard Statistics can be derived using this....

"""
Some standard statistics
.head()
.tail()
.sample()
.info()
.describe()
"""

# this is what we normally do when describing the data

df3.describe().transpose()

# conditional subsetting if you want to do

df[df['Height']>155]

# We can put more than one condition also in here

df[df['X']> 155 &  df['Y']>23]


# finding the range that is max - min

range_data = df['X'].max()-df['X'].min()

# finding the 95 percentile

value_95 = np.percentile(df['X'],95)

# finding the top five percentile which have content more than that

df[ df['X'] > value_95]

# showing the less attributes that is only which is required
df3[df3['Flavanoids'] >= 3.4975][['Ash','Alcohol','Magnesium']].mean()


# using the inplace value equal to true
df4.sort_values(by='BMI', inplace=True)

# dropping the column and with the axis=1 value and the inplace value is equal to the true
df.drop('Row ID', axis=1, inplace = True)

# subsetting the dataFrame using the rows and columns
# loc for getting the data through the Label
# iloc for getting the data using the index

df_subset = df.loc[[i for i in range(100)], ["X", "Y", "Z"]]








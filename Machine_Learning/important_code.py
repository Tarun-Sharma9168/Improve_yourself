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


#Finding the statisitics on the subset only that is

df_subset.describe()



#Using the unique functions 
unique_values = df['X'].unique()
number_unique_values=df['Y'].nunique()

#Using the Group by method and remember there is no existence of groupby function without the aggregation method


#Pyspark important 
from Pyspark.sql import SparkSession

spark=SparkSession.builder.appName('binary_class').getOrCreate()



#reading the dataset into the spark cluster inferSchema is really very necessary as it is important in Pyspark along with the header using the 
#spark function that is spark.read.csv just like the pandas we use generally
df=spark.read.csv('/FileStore/tables/classification_data.csv',inferSchema=True,header=True)


#best way to do the visualization you can use certain palettes
sns.pairplot(data=df,hue="Species",palette="Set2")
plt.show()



#getting the datavalues for certain rows using the label and of all the rows
features = df.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
 
 
# Finding the best  number of clusters for kmeans 

from sklearn.cluster import KMeans
wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)


plt.figure(figsize=(20,8))
plt.title("WCSS / K Chart", fontsize=18)
plt.plot(range(1,15),wcss,"-o")
plt.grid(True)
plt.xlabel("Amount of Clusters",fontsize=14)
plt.ylabel("Inertia",fontsize=14)
plt.xticks(range(1,20))
plt.tight_layout()
plt.show()


#Taking the logarithm if you want
df['Log-area']=np.log10(df['area']+1)


#Scatter plot using the dataset directly remember the paramters you can use
#visualizing the grid can be a good parameter as it enhances the visualization
for i in df.describe().columns[:-2]:
    df.plot.scatter(i,'Log-area',grid=True)


#Also the boxplot if you want  
df.boxplot(column='Log-area',by='day')
df.boxplot(column='Log-area',by='month')

#There are different modes of preprocessing that we needs to do 
#Remember converting the machine learning categorical variables into the machine learning 
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,ShuffleSplit,KFold

#Label Encoding any values 
#Simple method that is fit with the categorical variables
enc = LabelEncoder()
enc.fit(df['month'])


df['month_encoded']=enc.transform(df['month'])
df.head()
#TRanformation is necessary after fitting the data


#Dropping the data if needed and after that train test split into certainratio
X_data=df.drop(['area','Log-area','month','day'],axis=1)
y_data=df['Log-area']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)


#Some important models along with the other things which is necessary along with the models 
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
scaler = StandardScaler()

#defining of the Parameter
# Parameter grid for the Grid Search
param_grid = {'C': [0.01,0.1,1, 10], 'epsilon': [10,1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

#there are two paramters involved in the svm where one is to maximize the margin of the hyperplane and 
#other penalizes the data points if wronlgy classifies and affect the hyperplane buuilding 

grid_SVR = GridSearchCV(SVR(),param_grid,refit=True,verbose=0,cv=5)
grid_SVR.fit(scaler.fit_transform(X_train),scaler.fit_transform(y_train))


#getting the best paramters 
print("Best parameters obtained by Grid Search:",grid_SVR.best_params_)

#Prediction is also very important 
a=grid_SVR.predict(X_test)
print("RMSE for Support Vector Regression:",np.sqrt(np.mean((y_test-a)**2)))

#importing the RandomForestRegressor here
from sklearn.ensemble import RandomForestRegressor

param_grid = {'max_depth': [5,10,15,20,50], 'max_leaf_nodes': [2,5,10], 'min_samples_leaf': [2,5,10],
             'min_samples_split':[2,5,10]}
grid_RF = GridSearchCV(RandomForestRegressor(),param_grid,refit=True,verbose=0,cv=5)
grid_RF.fit(X_train,y_train)


#Sometimes simple network is also important with keras model

from keras.models import Sequential
import keras.optimizers as opti
from keras.layers import Dense,Activation,MaxPool1D,Dropout
from keras.layers import Conv1D,Conv2D,LSTM,GRU,Embedding



#Building a simple sequential model 
#it is very important to understand that it is not much difficult 
model=Sequential()
model.add(Dense(100,input_dim=12))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()


#Neural Network much more than the skeleton of layers some other things are also important 
learning_rate=0.001
optimizer = opti.RMSprop(lr=learning_rate)
model.compile(optimizer=optimizer,loss='mse')


#fitting the model to the data
data=X_train
target = y_train
model.fit(data, target, epochs=100, batch_size=10,verbose=0)

#Building the REC Curve for the neural network 
rec_NN=[]
for i in range(tol_max):
    rec_NN.append(rec(a,y_test,i))

plt.figure(figsize=(5,5))
plt.title("REC curve for the Deep Network\n",fontsize=15)
plt.xlabel("Absolute error (tolerance) in prediction ($ha$)")
plt.ylabel("Percentage of correct prediction")
plt.xticks([i for i in range(0,tol_max+1,5)])
plt.ylim(-10,100)
plt.yticks([i*20 for i in range(6)])
plt.grid(True)
plt.plot(range(tol_max),rec_NN)


#i dont know but we use this generally to make the graph,plots within the notebooks itself 
#or making the matplotlib inline 
#%matplotlib inline sets the backend of matplotlib to the 'inline' backend: 
#With this backend, the output of plotting commands is displayed inline within 
#frontends like the Jupyter notebook, directly below the code cell that produced it
#The resulting plots will then also be stored in the notebook document




#checking the shape
df.shape
df.info()


#If you want the value count of your values you can do this using the function value_counts()
# which is very important 

#we are counting the credit policy status like how many loans have approved and how many are not 
#using the function value_counts()
print("Follwoing is a breakup of credit approval status. 1 means approved credit, 0 means not approved.")
print(df['credit.policy'].value_counts())



#EXploratory Data Analyis
# you are simply taking the people credit score and want to make a comparison
# it is very important histogram code which does not 
#Building the histogram 
df[df['credit.policy']==1]['fico'].plot.hist(bins=30,alpha=0.5,color='blue', label='Credit.Policy=1')

df[df['credit.policy']==0]['fico'].plot.hist(bins=30,alpha=0.5, color='red', label='Credit.Policy=0')

plt.legend(fontsize=15)

plt.title ("Histogram of FICO score by approved or disapproved credit policies", fontsize=16)

plt.xlabel("FICO score", fontsize=14)


import seaborn as sns
#Using the Seaborn Boxplot Feature simple and effective and easy to use
#plotting the boxplot using the credit 
#it is even given that the risky people got the higher interest from the bank which is usual because bank doesnot 
#believe in those people
#the plot not only includes the Plot Also the lables including the X label ,Ylable ,legend the size 
#And the main thing is the Title of the plot
sns.boxplot(x=df['credit.policy'],y=df['int.rate'])

plt.title("Interest rate varies between risky and non-risky borrowers", fontsize=15)

plt.xlabel("Credit policy",fontsize=15)

plt.ylabel("Interest rate",fontsize=15)




#it is very important plot where the reason of the loan is shown in which another attribute not fully paid attribute is
#act as a hue
#As Count Plot is very important measure which deals with the count values so 
#Sns that is seaborn gives you very nice code and the view after that
plt.figure(figsize=(10,6))

sns.countplot(x='purpose',hue='not.fully.paid',data=df, palette='Set1')

plt.title("Bar chart of loan purpose colored by not fully paid status", fontsize=17)

plt.xlabel("Purpose", fontsize=15)




#We are trying to find a trend between the fico score and the interest rate that is 
#why it is called the joint plot because it is the combination of the distribution and the scatterplot
#There are some nice plot that is very attractive and the hybrid plots containing more than one plot
#In a single plot and joint plot from the sns gives the field of that 

sns.jointplot(x='fico',y='int.rate',data=df, color='purple', size=12)



#Another one is coming which is lmplot which fits the regression line between the attributes as well
#as draw the scatter plot between the two it is computationally more intensive than the regplot which 
#only fits the simple linear regression line and it is computationally less expensive and on the other
#hand the lmplot is using the regplot as well as the facetgrid plt.figure(figsize=(14,7))

sns.lmplot(y='int.rate',x='fico',data=df,hue='credit.policy',
           col='not.fully.paid',palette='Set1',size=6)



#We have to convert the categorical variables using the get dummies by dropping the first_one 
#it is the best way to do this
df_final = pd.get_dummies(df,['purpose'],drop_first=True)



#Training a decision tree classifier using the criterion as a ginny and the max_depth parameter as None

#we use the gini as the criterion
dtree = DecisionTreeClassifier(criterion='gini',max_depth=None)




#Describing the result is very important is very important so 
#classification report is very well known way to represent the result 
#Classification report and the confusion can be a measure but it is good to start with accuracy as measure otherwise not
from sklearn.metrics import classification_report,confusion_matrix


#again the same way that is y_test,y_pred pair is used 
print(classification_report(y_test,predictions))


#Similarly printing the Confusion matrix is also really important 
cm = confusin_matrix(y_test,y_pred)
print(cm)
print ("Accuracy of prediction:",round((cm[0,0]+cm[1,1])/cm.sum(),3))


#Training the Random Forest Classifier is an art and you should know that because 
#these ensemble classifiers are also very important Remember their parameters is also very important 
from sklearn.ensemble import RandomForestlassifier

#you took 600 as the number of estimators is nothing but the numberof trees
rfc = RandomForestClassifier(n_estimators=600)

#Simply fitting the the value using the Training Data
rfc.fit(X_train, y_train)

#getting the columns of the dataset 
df.columns

#Making the boxplot for all the variables
#this time not the commmon one but we are using the sns boxplot which is really very important

#Most easy way to plot the data .You can mention just the attribute name if you are giving the data
for i in range(len(l)-1):
    sns.boxplot(x='TARGET CLASS',y=l[i], data=df)
    plt.figure()
    
    


#Scaling the features which is necessary using the sklearn.preprocessing library 



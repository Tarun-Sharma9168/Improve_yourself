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
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#transform the data is also that is fit and the transform the data 
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


#getting the training data except the label value 
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()




#fitting the Kneighbbors CLassifier model which has the important parameter that is n_neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


#Finding the misclassification error
#finding the misclassification error rate
print("Misclassification error rate:",round(np.mean(pred!=y_test),3))





#Finding the K using the Elbow method where the plot is between thw miclassification rate and the 
#value of k
error_rate = []
# Will take some time
for i in range(1,60): 
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
    
    
#plot this value and you get the easy idea what you gonna do to the value of  K
#We are going to draw a very nice plot in which we are using different colours ,different types of lines
#title and the variety in the fontsize and all the other facilities that are there 
#And also generating the k value using the range function
plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.title('Error Rate vs. K Value', fontsize=20)
plt.xlabel('K',fontsize=15)
plt.ylabel('Error (misclassification) Rate',fontsize=15)    

    

#Sometimes splitting the data is needed in machine learning and that is using the simple split functions

names = names.split(sep=' ')
names    


#there are facilities when you are importing the data that is you can replace the null values and the 
#including the index cols or not or the names of the columns or not 

df = pd.read_csv('../Datasets/hypothyroid.csv',index_col=False,names=names,na_values=['?'])


#checking if there is a null in each column and if it is there how much is there
#very simple and intuitive
df.isnull().sum()



# Simple tranformation functions apply to rows these transformation can be anything and you can apply 
# using the apply method

def class_convert(response):
    if response=='hypothyroid':
        return 1
    else:
        return 0
    
    
#applying a particular function to a column usng the apply function
#apply takes the name of theb function that you want to apply on that particular row 
df['response']=df['response'].apply(class_convert)    



#Setting the pairplot with lot of differnt functionalities that are there
#plotting the pairplot using the seaborn
#in this you have to give the dataset only or column names if you want to build on some attributes only
#diag_kws and the plot_kws are the two differnt parts that you can separately define according to your own will
#how you want to see the plot 
sns.pairplot(data=df[df.columns[1:]],diag_kws={'edgecolor':'k','bins':25},plot_kws={'edgecolor':'k'})
plt.show()



#getting the sample out of the data same as like head but i think sample is random in the python 
#But i am not sure
#not only heads but the sample function also used to get the sample from the dataframe
df_dummies.sample(10)



#using The Logistics regression which uses the sigmoid function to map the values between 0 and 1 and inputs from the regression line
#map into the S Shaped Curve
from sklearn.linear_model import LogisticRegression
#Remember to mention the penalty and also the solver you want to use in solving the LogisticRegression 
#the parameters you can choose that is penalty and the solver for the parameters i think 
clf1 = LogisticRegression(penalty='l2',solver='newton-cg')
clf1.fit(X_train,y_train)
#you can map the probabilities value that are coming using the predict_proba method 
#it is very important to see this raw matching of the output
#It is nice as it is called by the classifier and it predicts the prob value
#taking the prob_threshold that is generally use in the logit function
prob_threshold = 0.5
prob_df=pd.DataFrame(clf1.predict_proba(X_test[:10]),columns=['Prob of NO','Prob of YES'])
#It is little bit tricky but nothing more difficult 
prob_df['Decision']=(prob_df['Prob of YES']>prob_threshold).apply(int)
prob_df 


#Creating the generalized linear model which has some requirementts including the formula which 
#Symbolises on how many variables the target value is dependent
import statsmodel.formula.api as smf
import statsmodels.api as sm
formula = 'response ~ ' + '+'.join(df_dummies.columns[1:])
formula
model = smf.glm(formula = formula, data=df_dummies, family=sm.families.Binomial())



#CountPlot using the Sns Style that is in the modern way
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')

#using hue if you want to groupby plots on certain attribute her attribute is the sex
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')



#Building a good quality histogram is necesary by using some of the important variables 
plt.xlabel("Age of the passengers",fontsize=18)
plt.ylabel("Count",fontsize=18)
plt.title("Age histogram of the passengers",fontsize=22)
train['Age'].hist(bins=30,color='darkred',alpha=0.7,figsize=(10,6))


#Now the Boxplot in the Sns Style  
#Choose differnt pallete values 
plt.figure(figsize=(12, 10))
plt.xlabel("Passenger Class",fontsize=18)
plt.ylabel("Age",fontsize=18)
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')



#Convert the features into the numerical values using the get_dummies
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)



#sometimes the concatenation is needed which is there with pandas 
train.drop(['Sex','Embarked'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model=keras.Sequential(layers.Dense(2,activation='relu',name='layer1'),
                       layers.Dense(3,activation='relu',name='layer2'),
                       layers.Dense(4,name='layers3'))

# As the dimension is 2d so the input that is required is 2d
# Call model on a test input
x = tf.ones((3, 3))
#the input is x here
y = model(x)


#we can create layer by layer
# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))

'''
A Sequential model is not appropriate when:
Your model has multiple inputs or multiple outputs
Any of your layers has multiple inputs or multiple outputs
You need to do layer sharing
You want non-linear topology (e.g. a residual connection, a multi-branch model)
'''


#it is very important 
#We can pass the list as well of the Dense layer 
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)



#We can access all the layers using the 
model.layers


#We can also create the Sequential model using the add method Model 
model=keras.Sequential()
model.add(Dense(2,name='layer1',activation='relu'))
model.add(Dense(5,activation='relu',name='layer2'))
model.add(Dense(1,activation='relu',name='layer3'))


#there is pop method is also there 
model.pop()
print(len(model.layers))


#checking the model weights is essential
model.weights


#simple model 
model=keras.Sequential()
model.add(keras.Input(shape=(250,250,3)))
model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))


#Getting the model summary
model.summary()

#Again building a sequential model
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))

#getting the model summary
model.summary()

# Now that we have 4x4 feature maps, time to apply global max pooling.
model.add(layers.GlobalMaxPooling2D())

# Finally, we add a classification layer.
model.add(layers.Dense(10))



#getting the feature extracted from layers one by one 
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)
# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)


#getting features from the particular layer by using the name
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)
# Call feature extractor on test input.
# the dataset of only one image 
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)



#very simple code for transfer learning that is 
model = keras.Sequential([
    keras.Input(shape=(784)),
    layers.Dense(32, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(10),
])

# Presumably you would want to first load pre-trained weights.
model.load_weights(...)

# Freeze all layers except the last one.
for layer in model.layers[:-1]:
  layer.trainable = False

# Recompile and train (this will only update the weights of the last layer).
model.compile(...)
model.fit(...)



#Using another model as well and make use of it
base_model=keras.applications.Xception(weights='imagenet',include_top=False,pooling='avg')
base_model.trainable=False
model=keras.Sequential([
    base_model,
    layers.Dense(1000),
])


#Compile and train
model.compile(...)
model.fit(...)


#Some hints for the Functional APi of the Models using keras
'''
The Keras functional API is a way to create models that are more flexible than the tf.keras.Sequential API. 
The functional API can handle models with non-linear topology, shared layers, and even multiple inputs or outputs.
The main idea is that a deep learning model is usually a directed acyclic graph (DAG) of layers. So the functional
API is a way to build graphs of layers.
'''


'''
The shape of the data is set as a 
784-dimensional vector. The batch size is always omitted since only the shape of each sample is specified.
'''
#If, for example, you have an image input with a shape of (32, 32, 3), you would use:
img_inputs=keras.Input(shape=(32,32,3))


#First the input is created that is remember nn should be a DAG
inputs=keras.Input(shape=(784,))
dense = layers.Dense(64, activation="relu")
x = dense(inputs)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")


#it is the general way 
model.summary()


#you can also plot the model as a graph
#very beautiful architectural plot will be visible
keras.utils.plot_model(model,"my_first_model.png")


#there is a paramter in this function that is show_shapes=True which is really very important 
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)


#loading the data for neural network 
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

#three things while compiling the model that is loss function,optimizer and the metrics 
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

#while fitting the batch_size epochs and validation_split
history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])


#saving the model
model.save("path_to_my_model")
del model
# Recreate the exact same model purely from the file:
model = keras.models.load_model("path_to_my_model")


#defining the encoder and the decoder
encoder_input = keras.Input(shape=(28,28,1),name='img')
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

#Now the decoder module
x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

#Autoencoder full
autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()

'''
As you can see, the model can be nested: a model can contain sub-models (since a model is just like a layer).
A common use case for model nesting is ensembling.
For example, here's how to ensemble a set of models into a single model that averages their predictions:
'''

def get_model():
    inputs=keras.Input(shape=(28,28,1))
    outputs=layers.Dense(1)(inputs)
    return keras.Model(inputs=inputs,outputs=outputs)

model1=get_model()
model2=get_model()
model3=get_model()

inputs=keras.Input(shape=(28,28,1))
y1=model1(inputs)
y2=model2(inputs)
y3=model3(inputs)

outputs=layers.average([y1,y2,y3])
ensemble_model = keras.Model(inputs,outputs)


#Multiple inputs and the Multiple outputs
num_tags = 12  # Number of unique issue tags
num_words = 10000  # Size of vocabulary obtained when preprocessing text data
num_departments = 4  # Number of departments for predictions

title_input = keras.Input(
    shape=(None,), name="title"
)  # Variable-length sequence of ints
body_input = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
tags_input = keras.Input(
    shape=(num_tags,), name="tags"
)  # Binary vectors of size `num_tags`

# Embed each word in the title into a 64-dimensional vector
title_features = layers.Embedding(num_words, 64)(title_input)
# Embed each word in the text into a 64-dimensional vector
body_features = layers.Embedding(num_words, 64)(body_input)

# Reduce sequence of embedded words in the title into a single 128-dimensional vector
title_features = layers.LSTM(128)(title_features)
# Reduce sequence of embedded words in the body into a single 32-dimensional vector
body_features = layers.LSTM(32)(body_features)

# Merge all available features into a single large vector via concatenation
x = layers.concatenate([title_features, body_features, tags_input])

# Stick a logistic regression for priority prediction on top of the features
priority_pred = layers.Dense(1, name="priority")(x)
# Stick a department classifier on top of the features
department_pred = layers.Dense(num_departments, name="department")(x)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred],
)


#Plotting this model gives you the immense pleasure to see what is there with you
keras.utils.plot_model(model,"plot_multiple_input_multiple_output",show_shapes=True)


#We can have the weighted loss for different outputs
#this is very important and   required to be done 
'''
When compiling this model, you can assign different losses to each output.You can even assign
different weights to each loss -- to modulate their contribution to the total training loss.
'''


# a toy resnet model for seeing how to build the skip connections
# And leverage the power of the functional API

inputs=keras.Input(shape=(32,32,3), name="img")
x=layers.Conv2D(32,3,activation="relu")(inputs)
x=layers.Conv2D(64,3,activation="relu")(inputs)
block_1_output=layers.MaxPooling2D(3)(x)


#Very nice code just see
x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])


x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name="toy_resnet")
model.summary()

model.save("toy_resnet")

keras.utils.plot_model(model,"toy_resnet_model.png",show_shapes=True)



#training the model is as easy as possible dont worry
(x_train,y_train),(x_test,y_test)=keras.datasets.cifar10.load_data()

x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0
y_train=y_train.utils.to_categorical(y_train,10)
y_test=y_test.utils.to_categorical(y_test,10)

#using the categorical cross_entropy
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["acc"],
)
# We restrict the data to the first 1000 samples so as to limit execution time
# on Colab. Try to train on the entire dataset until convergence!
model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=1, validation_split=0.2)


# Shareable layers which are really very important if needed
shared_embedding=layers.Embedding(1028,128)
# Variable-length sequence of integers
text_input_a = keras.Input(shape=(None,), dtype="int32")

# Variable-length sequence of integers
text_input_b = keras.Input(shape=(None,), dtype="int32")

# Reuse the same layer to encode both inputs
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)





#Using the vgg model and the nodes of dag that is the layers in between
vgg19 = tf.keras.applications.VGG19()

feature_list=[layer.output for layer in vgg19.layers]

#using this feature list we are creating another model that gives us intermediate activations
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=feature_list)

img = np.random.random((1, 224, 224, 3)).astype("float32")
extracted_features = feat_extraction_model(img)





# Reserve 10,000 samples for validation sets in the keras for training a neural network
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

#compiling the model with the RMSprop and the SparseCategoricalCrossentropy,SparseCategoricalAccuracy
model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)   

#fitting the data
print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)


#the returned history objects holds a record of the loss values and metric values during training
history.history

#evaluate the results
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)



#Remember metrics should always be a list so keep it in mind and we need this 
#when we use the multiple inputs and the multiple outputs model
def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model



'''
Optimizers Available:
SGD
Adam
RMSprop
'''

'''
Losses Available:
MeanSquaredError()
KlDivergence()
CosineSimilarity()
etc
'''

'''
Metrics:
AUC 
Precision()
Recall()
'''

def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))


model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(), loss=custom_mean_squared_error)

# We need to one-hot encode the labels to use MSE
y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)



#loss function with some other paramters as well you have to define a class
#that is a custommse if you want to make you have to define the ___init__ function and the call function
#if you want to use more than one parameter..........
class CustomMSE(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor


model = get_uncompiled_model()
#compiling with the custommse
model.compile(optimizer=keras.optimizers.Adam(), loss=CustomMSE())

y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)




#Building a custome metrics that is really very important 
class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)


model = get_uncompiled_model()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[CategoricalTruePositives()],
)
model.fit(x_train, y_train, batch_size=64, epochs=3)



#Adding the ActivityRegularizer
class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs) * 0.1)
        return inputs  # Pass-through layer.


inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)

# Insert activity regularization as a layer
x = ActivityRegularizationLayer()(x)

x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# The displayed loss will be much higher than before
# due to the regularization component.
model.fit(x_train, y_train, batch_size=64, epochs=1)




#Adding the MetricLogging Layer
class MetricLoggingLayer(layers.Layer):
    def call(self, inputs):
        # The `aggregation` argument defines
        # how to aggregate the per-batch values
        # over each epoch:
        # in this case we simply average them.
        self.add_metric(
            keras.backend.std(inputs), name="std_of_activation", aggregation="mean"
        )
        return inputs  # Pass-through layer.


inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)

# Insert std logging as a layer.
x = MetricLoggingLayer()(x)

x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
model.fit(x_train, y_train, batch_size=64, epochs=1)




#when you add the loss using the add_loss and the metric using the add_metric
#then you do not have to add the loss and metric when you are compiling the model

class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy as a metric and add it
        # to the layer using `self.add_metric()`.
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)


  
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing the dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[ ]:


#loading the datset
from pandas import read_csv
diabetes_data=pd.read_csv('diabetes.csv')


# In[ ]:


diabetes_data.head()


# In[ ]:


#find the rows and columns
diabetes_data.shape


# In[ ]:


#describe the data
diabetes_data.describe


# In[ ]:


#count the examples of diabetic and non diabetic
diabetes_data['Outcome'].value_counts()


# In[ ]:


#separating the outcome 
diabetes_data.groupby('Outcome').mean()


# In[ ]:


#separting the data
X=diabetes_data.drop(columns='Outcome',axis=1)
print(X)


# In[ ]:


Y=diabetes_data['Outcome']
print(Y)


# In[ ]:


#Data Standaization
scaler= StandardScaler()


# In[ ]:


scaler.fit(X)


# In[ ]:


standardized_data=scaler.transform(X)


# In[ ]:


print(standardized_data)


# In[ ]:


X=standardized_data
Y=diabetes_data['Outcome']


# In[ ]:


print(X)


# In[ ]:


print(Y)


# In[ ]:


#spliting training and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[ ]:


print(X.shape,X_train.shape,X_test.shape)


# In[ ]:


print(Y.shape,Y_train.shape,Y_test.shape)


# In[ ]:


#Model
classifier=svm.SVC(kernel='linear')


# In[ ]:


classifier.fit(X_train,Y_train)


# In[ ]:


#Model Evalution
#Accuracy of training data
X_train_prediction=classifier.predict(X_train)
training_prediction_accuaracy=accuracy_score(X_train_prediction,Y_train)


# In[ ]:


print('Accuracy of taining data:',training_prediction_accuaracy)


# In[ ]:


#Accuracy of training data
X_test_prediction=classifier.predict(X_test)
testing_prediction_accuaracy=accuracy_score(X_test_prediction,Y_test)


# In[ ]:


print('Accuracy of testing data:',testing_prediction_accuaracy)


# In[ ]:


#Prediction of the model
input_data=(1,85,66,29,0,26.6,0.351,31)
#changing the input data into numpy
input_data_array=np.asarray(input_data)

input_data_reshaped=input_data_array.reshape(1,-1)

#refitting x training data
std_data=scaler.transform(input_data_reshaped)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)

if(prediction[0]==0):
    print('The person is non diabetic')
else:
    print('The person is diabetic')


# In[ ]:





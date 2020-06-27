import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
titanic_data=pd.read_csv('train.csv')
print(titanic_data)
print(titanic_data.head(10))
print(titanic_data.shape)
print("total no of passengers:"+str(len(titanic_data.index)))
sns.countplot(x="Survived",hue="Age",data=titanic_data)
sns.countplot(x="Survived",data=titanic_data)
titanic_data["Age"].plot.hist()
titanic_data["Fare"].plot.hist()
titanic_data["Pclass"].plot.hist()


print(titanic_data.info())
print(titanic_data.isnull())
print(titanic_data.isnull().sum())
titanic_data.boxplot(by ='Pclass', column =['Age'], grid = False) 
print(titanic_data.head())
titanic_data.drop('Cabin',axis=1,inplace=True)
print(titanic_data.head())
print(titanic_data.columns)
print(titanic_data.dropna(inplace=True))
print(titanic_data.info())
sns.heatmap(titanic_data.isnull(), xticklabels=False, yticklabels=False)
print(titanic_data.isnull().sum())#check the overall null values
sex=pd.get_dummies(titanic_data['Sex'],drop_first='True')
print(sex.head(5))
embarked=pd.get_dummies(titanic_data['Embarked'],drop_first='True')
print(embarked.head(20))
pclass=pd.get_dummies(titanic_data['Pclass'],drop_first='True')
print(pclass)
print(pclass.head(10))
titanic_data=pd.concat([titanic_data,sex,embarked,pclass],axis=1)
titanic_data.drop(['Name','Sex','Embarked','PassengerId','Ticket'],axis=1,inplace=True)
print(titanic_data)
print(titanic_data.head(5))
titanic_data.drop('Pclass',axis=1,inplace=True)
print(titanic_data.head(5))
#training
x=titanic_data.drop("Survived",axis=1)
print(x)
y=titanic_data["Survived"]
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)    
y_pred=reg.predict(x_test)
print(y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)

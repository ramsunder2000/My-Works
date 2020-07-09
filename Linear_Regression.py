import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model

df=pd.read_csv("homeprice.csv")
print(df)
reg=linear_model.LinearRegression()     # Linear Regression
reg.fit(df[['Area']],df.Price)          # Linear Regression
x = np.linspace(2000, 4500, 20)         #To display the graph in applet like window
plt.axis([2000, 4500, 500000, 750000]) # Can adjust x and y axis , but the linear equation will change      
plt.plot(x, np.sin(x),color='green')
plt.xlabel('area(in Sqr feet)')
plt.ylabel('Price in US$')
plt.scatter(df.Area,df.Price,color='red',marker='+')
plt.plot(df.Area,reg.predict(df[['Area']]),color='blue')
plt.show()

print("Predicted value of 3400 Sq feet=",reg.predict([[3400]]))            #2D array expected , thats why 2 square brackets
print("Coefficient of 3400 Sq feet=",reg.coef_)
print("Interception of 3400 Sq feet=",reg.intercept_)

d=pd.read_csv('areas.csv')
print(d.head(3))
p=reg.predict(d)
d['Prices']=p
d.to_csv('prediction.csv',index=False)

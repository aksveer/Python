import pandas as pd;
import quandl
import math
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import numpy as np


df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
df['PCT_Change']= (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0

df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
df.dropna(inplace=True)

#print(forecast_col)

df['lable'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

x= np.array(df.drop(['lable'],1))
y= np.array(df['lable'])

x = preprocessing.scale(x)

x_train , x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.2)

#print(df.isna().sum())

clf = LinearRegression()

clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)

print(accuracy)
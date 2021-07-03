# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:00:08 2021

@author: 52pun
"""
from keras.models import load_model

model = load_model("C:/Users/52pun/Desktop/Repos/Stock_Price_Prediction/python_script/saved_models/UnionBankOfIndia.h5")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


union_df = pd.read_csv("C:/Users/52pun/Desktop/Repos/Stock_Price_Prediction/Data/Bank_data/UNIONBANK_5Y.csv")

u_close = union_df["Close"]
u_close = u_close.dropna()
u_close.shape


scaler = MinMaxScaler(feature_range=(0,1))
tmp = scaler.fit(np.array(u_close).reshape(-1,1))
new_df = scaler.transform(np.array(u_close).reshape(-1,1))

 
print(new_df.shape)

training_size=int(len(new_df)*0.67)
test_size=len(new_df)-training_size
train_data,test_data=new_df[0:training_size,:],new_df[training_size:len(new_df),:1]

print(train_data.shape)
print(test_data.shape)

x_input=test_data[307:].reshape(1,-1)
print(x_input.shape)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        print(len(temp_input))
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(len(lst_output))
day_new=np.arange(1,101)
day_pred=np.arange(101,131)


plt.plot(day_new,scaler.inverse_transform(new_df[1132:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
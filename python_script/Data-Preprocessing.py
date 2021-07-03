# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 11:12:23 2021

@author: 52pun
"""

import pandas as pd
import matplotlib.pyplot as plt

bob_df = pd.read_csv("C:/Users/52pun/Desktop/Mini Project/Data/BANKBARODA_5Y.csv")
#print(bob_df)
#boi_df = pd.read_csv("C:/Users/52pun/Desktop/Mini Project/Data/Bank_data/BOI_5Y.csv")
pnb_df = pd.read_csv("C:/Users/52pun/Desktop/Mini Project/Data/Bank_data/PNB_5Y.csv")
sbi_df = pd.read_csv("C:/Users/52pun/Desktop/Mini Project/Data/Bank_data/SBIN_5Y.csv")
ubi_df = pd.read_csv("C:/Users/52pun/Desktop/Mini Project/Data/Bank_data/UNIONBANK_5Y.csv")

#print(boi_df)
print(pnb_df)
print(sbi_df)
print(ubi_df)
print(ubi_df.columns)

dates = bob_df["Date"]
bob_close = bob_df["Close"]
#boi_close = boi_df["Close"]
pnb_close = pnb_df["Close"]
sbi_close = sbi_df["Close"]
ubi_close = ubi_df["Close"]

print(f"DATES: {dates}")
print(bob_close)
print(ubi_close)
print(sbi_close)

'''

plt.plot(dates, bob_close, color="red")
plt.plot(dates, sbi_close, color="blue")
plt.plot(dates, pnb_close, color="orange")
#plt.plot(dates, sbi_close, color="black")
plt.plot(dates, ubi_close, color="yellow")

'''



def update_data(close):
    
    '''
    momentum = []
    change = []
    '''
    momentum = [0]*len(dates)
    change = [0]*len(dates)
    
    for i in range(1,len(dates)):
        if close[i]>close[i-1]:
            momentum[i] = 1
        else:
            momentum[i] = 0
        
        change[i] = ((close[i]-close[i-1])/close[i-1])
        
    print(len(momentum))
    print(len(close))
    print(len(change))
    new=pd.DataFrame({'Date': dates, 'Close': close, 'Momentum': momentum, 'Change': change})
    return new

bob_update = update_data(bob_close)
bob_update.to_csv('bob_update_df.csv', index=False, header=True)

print(bob_update)
new_df = bob_update["Close"]
new_df = new_df.dropna()
print(f"Shape----{new_df.shape}")

from numpy import asarray
from numpy import savetxt
# define data

data = asarray(new_df)
savetxt('original.csv', data, delimiter=',')


import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
new_df = scaler.fit_transform(np.array(new_df).reshape(-1,1))

print(new_df)


training_size=int(len(new_df)*0.67)
test_size=len(new_df)-training_size
train_data,test_data=new_df[0:training_size,:],new_df[training_size:len(new_df),:1]

print(train_data.shape)
print(test_data.shape)

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

time_step=100
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
'''
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
print(X_train.shape, X_test.shape)
'''


# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# define data



data = asarray(X_train)
savetxt('X_train.csv', data, delimiter=',')
data = asarray(Y_train)
savetxt('Y_train.csv', data, delimiter=',')
data = asarray(X_test)
savetxt('X_test.csv', data, delimiter=',')
data = asarray(Y_test)
savetxt('Y_test.csv', data, delimiter=',')
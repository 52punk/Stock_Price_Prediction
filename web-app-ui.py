# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:14:48 2021

@author: 52pun
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt





model = load_model("python_script/saved_models/UnionBankOfIndia.h5")


union_data = pd.read_csv("Data/Bank_data/UNIONBANK_5Y.csv")
boi_data = pd.read_csv("Data/Bank_data/BOI_5Y.csv")
bob_data = pd.read_csv("Data/BANKBARODA_5Y.csv")
sbi_data = pd.read_csv("Data/Bank_data/SBIN_5Y.csv")
pnb_data = pd.read_csv("Data/Bank_data/PNB_5Y.csv")

stocks = {'Union Bank of India': union_data, 'State Bank of India': sbi_data, 'Bank of India': boi_data, 'Bank of Baroda': bob_data, 'Punjab National Bank': pnb_data}
stocks_data = ('Union Bank of India', 'State Bank of India', 'Bank of India', 'Bank of Baroda', 'Punjab National Bank')





def choose_dataset(stocks, stocks_data):
    st.sidebar.subheader('Select the bank')
    stock = st.sidebar.selectbox( "", stocks_data, key='1' )
    check = st.sidebar.checkbox("Hide", value=True, key='1')
    #st.sidebar.write(check)
    for itr in stocks_data:
        if stock==itr:
            main_df=stocks[itr]
    return main_df, check, stock




def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)



def plot_predict(df):
    
    date = df["Date"]
    df = df["Close"]
    df = df.dropna()
    ori_df = df
    df.shape
    #print(f" {df}")
    scaler = MinMaxScaler(feature_range=(0,1))
    tmp = scaler.fit(np.array(df).reshape(-1,1))
    new_df = scaler.transform(np.array(df).reshape(-1,1))
    
    
    #print(new_df.shape)
    
    
    training_size=int(len(new_df)*0.67)
    test_size=len(new_df)-training_size
    train_data,test_data=new_df[0:training_size,:],new_df[training_size:len(new_df),:1]
    
    #print(train_data.shape)
    #print(test_data.shape)
    
    
    time_step=100
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    #print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    #print(X_train, Y_train)
    
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    #print(X_train.shape, X_test.shape)
    #print(X_train, Y_train)
    
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    #print(train_predict.shape, test_predict.shape)
    
    
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    
    look_back=100
    trainPredictPlot = np.empty_like(new_df)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back-2:len(train_predict)+look_back-2, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(new_df)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)-1:len(new_df)-3, :] = test_predict
    #ori_df = scaler.inverse_transform(new_df).reshape(-1,1)
    #print(f"Original data {ori_df}")
    ori_df = pd.DataFrame(ori_df)
    #print(f"ori {ori_df}")
    trainPredictPlot = pd.DataFrame(trainPredictPlot)
    testPredictPlot = pd.DataFrame(testPredictPlot)
    #print(f"To dataframe {ori_df}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date, y=ori_df["Close"], name="Original closing prices"))
    fig.add_trace(go.Scatter(x=date, y=trainPredictPlot[0], name="train_close"))
    fig.add_trace(go.Scatter(x=date, y=testPredictPlot[0], name="test_close"))
    fig.layout.update( xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    #fig.show()
    
    


def plot_forecast_data(df, days):
    u_close = df["Close"]
    u_close = u_close.dropna()
    u_close.shape
    
    
    scaler = MinMaxScaler(feature_range=(0,1))
    tmp = scaler.fit(np.array(u_close).reshape(-1,1))
    new_df = scaler.transform(np.array(u_close).reshape(-1,1))
    training_size=int(len(new_df)*0.67)
    test_size=len(new_df)-training_size
    train_data,test_data=new_df[0:training_size,:],new_df[training_size:len(new_df),:1]
    
    x_input=test_data[307:].reshape(1,-1)
    print(x_input.shape)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=100
    i=0
    while(i<days):
        
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
    
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    lst_output = scaler.inverse_transform(lst_output)
    lst_output = pd.DataFrame(lst_output)
    y_data = scaler.inverse_transform(new_df[1132:])
    y_data = pd.DataFrame(y_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=day_new, y=y_data[0], name="1"))
    fig.add_trace(go.Scatter(x=day_pred, y=lst_output[0], name="2"))
    fig.layout.update( xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    #plt.plot(day_new,scaler.inverse_transform(new_df[1132:]))
    #plt.plot(day_pred,scaler.inverse_transform(lst_output))



    

def plot_raw_data(data):
    
   	fig = go.Figure()
   	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
   	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
   	fig.layout.update( xaxis_rangeslider_visible=True)
   	st.plotly_chart(fig)
    	

def landing_ui():
    st.header("Welcome to Stock Price Predictor")
    st.write("Welcome to this site, it is still in its beta stage!")
    st.write("To see the data representation please uncheck the hide button in the sidebar")
    

if __name__ == "__main__":
    
    st.sidebar.subheader("Stock Market Predictor")
    st.sidebar.markdown("---")
    temp, check, name=choose_dataset(stocks, stocks_data)
    #print(temp)
    if not check:
        st.header(f"Analyzing {name}'s stock data")
        st.subheader("Raw Data")
        st.write(temp)
        
        
        st.subheader("Raw Data - Visualized")
        plot_raw_data(temp)
        st.subheader("Predicted data")
        plot_predict(temp)
        st.sidebar.subheader("Forecasted Data")
        forecast_check = st.sidebar.checkbox("try", value=False)
        if forecast_check:
            forecast = st.slider("Days to forecast",min_value=10,max_value=100,step=5)
            st.subheader("Forcasted data")
            plot_forecast_data(temp, forecast)
    else:
        landing_ui()

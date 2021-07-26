# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:14:48 2021

@author: 52pun
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler





union_model = load_model("python_script/saved_models/UnionBankOfIndia.h5")
boi_model = load_model("python_script/saved_models/BankOfIndia_model.h5")
bob_model = load_model("python_script/saved_models/BankOfBaroda.h5")
sbi_model = load_model("python_script/saved_models/StateBankOfIndia_model.h5")
pnb_model = load_model("python_script/saved_models/PunjabNationalBank.h5")

union_data = pd.read_csv("Data/Bank_data/UNIONBANK_5Y.csv")
boi_data = pd.read_csv("Data/Bank_data/BOI_5Y.csv")
bob_data = pd.read_csv("Data/BANKBARODA_5Y.csv")
sbi_data = pd.read_csv("Data/Bank_data/SBIN_5Y.csv")
pnb_data = pd.read_csv("Data/Bank_data/PNB_5Y.csv")

allmodels = {'Union Bank of India': union_model, 'State Bank of India': sbi_model, 'Bank of India': boi_model, 'Bank of Baroda': bob_model, 'Punjab National Bank': pnb_model}
stocks = {'Union Bank of India': union_data, 'State Bank of India': sbi_data, 'Bank of India': boi_data, 'Bank of Baroda': bob_data, 'Punjab National Bank': pnb_data}
stocks_data = ('Union Bank of India', 'State Bank of India', 'Bank of India', 'Bank of Baroda', 'Punjab National Bank')



n_steps = 30

def choose_dataset(stocks, stocks_data, allmodels):
    st.sidebar.subheader('Select the bank')
    stock = st.sidebar.selectbox( "", stocks_data, key='1' )
    check = st.sidebar.checkbox("Hide", value=True, key='1')
    
    #st.sidebar.write(check)
    for itr in stocks_data:
        if stock==itr:
            main_df=stocks[itr]
            model=allmodels[itr]
    return main_df, check, stock, model


def about_section():
    st.sidebar.subheader('Made By:')
    st.sidebar.markdown("Mrinab Dey")
    st.sidebar.markdown('[LinkedIn](https://www.linkedin.com/in/mrinabdey/) [Github](https://github.com/mrinabdey)', unsafe_allow_html=True)
    st.sidebar.markdown("Pankaj Kumar Sah")
    st.sidebar.markdown('[LinkedIn](https://www.linkedin.com/in/pankaj-sah-b7aa39186/) [Github](https://github.com/52punk)', unsafe_allow_html=True)
    

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)



def plot_predict(df, model, name):
    
    
    df = df.drop(["Open", "Low", "Adj Close", "Volume"], axis=1)
    df = df.dropna()
    Date = df["Date"]
    close = df["Close"]
    close = close.dropna()
    scaler = MinMaxScaler(feature_range=(0,1))
    tmp = scaler.fit(np.array(close).reshape(-1,1))
    new_df = scaler.transform(np.array(close).reshape(-1,1))
    
    training_size=int(len(new_df)*0.67)
    test_size=len(new_df)-training_size
    train_data,test_data=new_df[:training_size],new_df[training_size:]
    Date_train, Date_test = Date[:training_size], Date[training_size:]
    
    n_steps = 30
    time_step=n_steps
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    print(X_train.shape, X_test.shape)
    
    
    
    
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    print(train_predict.shape, test_predict.shape)
    
    from sklearn.metrics import mean_squared_error
    print(f'Train error - {mean_squared_error(train_predict, Y_train)*100}')
    print(f'Test error - {mean_squared_error(test_predict, Y_test)*100}')
    
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    X_train=X_train.reshape(-1, 1)
    X_test=X_test.reshape(-1, 1)
    close_train=scaler.inverse_transform(train_data)
    close_test=scaler.inverse_transform(test_data)
    close_train = close_train.reshape(-1)
    close_test = close_test.reshape(-1)
    prediction = test_predict.reshape((-1))
    
    trace1 = go.Scatter(
        x = Date_train,
        y = close_train,
        mode = 'lines',
        name = 'Data'
    )
    trace2 = go.Scatter(
        x = Date_test[n_steps:],
        y = prediction,
        mode = 'lines',
        name = 'Prediction'
    )
    trace3 = go.Scatter(
        x = Date_test,
        y = close_test,
        mode='lines',
        name = 'Ground Truth'
    )
    layout = go.Layout(
        title = name,
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Close"}
    )
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    

    st.plotly_chart(fig)
    #fig.show()
    
    


def plot_forecast_data(df, days, model, name):
    
    df = df.drop(["Open", "Low", "Adj Close", "Volume"], axis=1)
    df = df.dropna()
    Date = df["Date"]
    close = df["Close"]
    close = close.dropna()
    scaler = MinMaxScaler(feature_range=(0,1))
    tmp = scaler.fit(np.array(close).reshape(-1,1))
    new_df = scaler.transform(np.array(close).reshape(-1,1))
    
    
    
    test_data = close
    test_data = scaler.transform(np.array(close).reshape(-1,1))
    test_data = test_data.reshape((-1))
    
    def predict(num_prediction, model):
        prediction_list = test_data[-n_steps:]
        
        for _ in range(num_prediction):
            x = prediction_list[-n_steps:]
            x = x.reshape((1, n_steps, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[n_steps-1:]
            
        return prediction_list
        
    def predict_dates(num_prediction):
        last_date = df['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
        return prediction_dates
    
    num_prediction =days
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)
    forecast = forecast.reshape(1, -1)
    forecast = scaler.inverse_transform(forecast)
    forecast
    test_data = test_data.reshape(1, -1)
    test_data = scaler.inverse_transform(test_data)
    test_data = test_data.reshape(-1)
    forecast = forecast.reshape(-1)
    res = dict(zip(forecast_dates, forecast))
    date = df["Date"]
    trace1 = go.Scatter(
        x = date,
        y = test_data,
        mode = 'lines',
        name = 'Data'
    )
    trace2 = go.Scatter(
        x = forecast_dates,
        y = forecast,
        mode = 'lines',
        name = 'Prediction'
    )
    layout = go.Layout(
    title = name,
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
    )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    st.plotly_chart(fig)
    #fig.show()
    choose_date = st.selectbox("Date", forecast_dates)
    for itr in res:
        if choose_date==itr:
            res_price=res[itr]
    st.write(f"On {choose_date} the stock price will be Rs. {res_price}")

    
    
    
    



    

def plot_raw_data(data):
    
   	fig = go.Figure()
   	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
   	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
   	fig.layout.update( xaxis_rangeslider_visible=True)
   	st.plotly_chart(fig)
    	

def landing_ui():
    st.header("Welcome to Stock Price Predictor")
    st.write("")
    st.write("")
    st.write("Welcome to this site")
    st.write("As the model is trained with data having time steps of 30 days so it will give its best results for a forecast till 30days ")
    st.write("")
    st.write("To see the data representation please uncheck the hide button in the sidebar")
    

if __name__ == "__main__":
    
    st.sidebar.subheader("Stock Market Predictor")
    st.sidebar.markdown("---")
    temp, check, name, model=choose_dataset(stocks, stocks_data, allmodels)
    #about_section()
    #print(temp)
    if not check:
        st.header(f"Analyzing {name}'s stock data")
        st.subheader("Raw Data")
        st.write(temp)
        
        
        st.subheader("Raw Data - Visualized")
        plot_raw_data(temp)
        st.subheader("Predicted data")
        plot_predict(temp, model, name)
        st.sidebar.subheader("Forecasted Data")
        forecast_check = st.sidebar.checkbox("See the results", value=False)
        about_section()
        if forecast_check:
            forecast = st.slider("Days to forecast",min_value=30,max_value=100,step=5)
            st.subheader("Forcasted data")
            
            plot_forecast_data(temp, forecast, model, name)
    else:
        landing_ui()

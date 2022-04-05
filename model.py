import yfinance as yf
import pandas as pd
from datetime import date
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

def stock_predict(v, days):
    # Get the stock data
    today = date.today()
    today = today.strftime("%Y-%m-%d")
    start_date = datetime.datetime.now() - datetime.timedelta((250*int(days)/100)+int(days))
    start_date = start_date.strftime("%Y-%m-%d")
    df = yf.download(v, start=start_date, end=today)
    df.reset_index(inplace=True)
    print(df.head())

    # remove irrelevant data
    df = df[['Adj Close']]

    # a variable for predicting 'n' days 
    forecast_out = int(days)

    # create another column (the dependent variable)
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out)

    # create the independent data set (X)
    X = np.array(df.drop(['Prediction'],1))

    # remove the last 'n' rows
    X = X[:-forecast_out]

    # create the dependent data set (Y)
    y = np.array(df['Prediction'])

    # remove the last 'n' rows
    y = y[:-forecast_out]

    # split the data into 90% training and 10%testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # create and train support vector machine (Regressor)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train) 

    # testing model: score return the coefficient of determination of R^2 of the prediction
    svm_confidence = svr_rbf.score(x_test, y_test)
    print('svm_confidence: ', svm_confidence)

    # create and train the linear regression model
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # testing the linear regression model
    lr_confidence = lr.score(x_test, y_test)
    print('lr_confidence: ', lr_confidence)

    # Set x_forecast equal to the last 30 rows of the original data set from Adj Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]

    # Print linear regression model prediction for the next 'n' days
    lr_prediction = lr.predict(x_forecast)
    print()

    # Print support vector regressor model prediction for the next 'n' days
    svm_prediction = svr_rbf.predict(x_forecast)
    print(svm_prediction)

    n_date = []
    for x in range(1,int(days)+1):
        temp_date = datetime.datetime.now() + datetime.timedelta(x)
        temp_date = temp_date.strftime("%Y-%m-%d")
        n_date.append(temp_date)
    
    df = pd.DataFrame(list(zip(n_date, svm_prediction)), columns =['Date', 'Prediction'])
    print(df.head())
    return df
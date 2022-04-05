import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
from datetime import date
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash.exceptions import PreventUpdate
import numpy as np
import urllib.request as ur
from model import stock_predict

app = dash.Dash(__name__)
server = app.server
code = ""

app.layout = html.Div(children=[html.Div(
    [
        html.P("Welcome to the Stock Dash App!", className="start"),
        html.Div([
            html.Label(['Input stock code: ',html.Br()]),
            dcc.Input(id='stock-code', type='text'),
            html.Button('Submit', id='submit-val', n_clicks=0)
        ]),
        html.Div([
            # Date range picker input
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=date(1995, 8, 5),
                max_date_allowed=date(2090, 8, 25),
                start_date=date(2021, 8, 5),
                end_date=date(2023, 8, 25)
            )
        ]),
        html.Div([
            # Stock price button
            html.Button('Stock price', id='stock-val', n_clicks=0),
            # Indicators button
            html.Button('Indicators', id='indicators-val', n_clicks=0),
            # Number of days of forecast input
            dcc.Input(id='number-days',type='text'),
            # Forecast button
            html.Button('Forecast', id='forecast-val', n_clicks=0)
        ]),
    ], className="nav"), 
    html.Div(
    [
        html.Div(children=[
            html.Img(id='logo', style={'height':'100px', 'width':'100px'}),
            html.Div(id='company_name')
        ], id='header'),
        html.Div(id='description', className="decription_ticker"),
        html.Div([
            dcc.Graph(id='stock_graph') 
        ], id="graphs-content", style={'display':'none'}),
        html.Div([
            dcc.Graph(id='indicator_graph')
        ], id="main-content", style={'display':'none'}),
        html.Div([
            dcc.Graph(id='forecast_graph')
        ], id="forecast-content", style={'display':'none'})
    ], className="content")
])


@app.callback(
    [Output('logo', 'src'),
    Output('company_name', 'children'),
    Output('description', 'children')],
    [Input('submit-val', 'n_clicks')],
    [State('stock-code', 'value')]
)

def update(n_clicks, v):
    if v==None:
        raise PreventUpdate
    ticker = yf.Ticker(v)
    inf = ticker.info
    df = pd.DataFrame().from_dict(inf, orient="index").T
    return [df['logo_url'].iat[0], df['shortName'].iat[0], df['longBusinessSummary'].iat[0]]

@app.callback(
    [Output('stock_graph', 'figure'),
    Output('graphs-content', 'style')],
    [Input('stock-val', 'n_clicks')],
    [State('date-picker', 'start_date'),
    State('date-picker', 'end_date'),
    State('stock-code', 'value')]
)

def stock_figure(n_clicks, start_date, end_date, v):
    if v==None:
        raise PreventUpdate
    
    if(n_clicks > 0):
        style = {'display':'block'}
    df = yf.download(v, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    fig = get_stock_price_fig(df)
    return [fig, style]

def get_stock_price_fig(df):
    figure= px.line(df,
            x= 'Date', 
            y= ['Open','Close'],
            title="Closing and Opening Price vs Date")
    return figure

@app.callback(
    [Output('indicator_graph', 'figure'),
    Output('main-content', 'style')],
    [Input('indicators-val', 'n_clicks')],
    [State('date-picker', 'start_date'),
    State('date-picker', 'end_date'),
    State('stock-code', 'value')]
)

def indicator_figure(n_clicks, start_date, end_date, v):
    if v==None:
        raise PreventUpdate
    
    if(n_clicks > 0):
        style = {'display':'block'}
    df = yf.download(v, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    fig = get_more(df)
    return [fig, style]

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    figure= px.line(df,
            x= 'Date', 
            y= 'EWA_20',
            title="Exponential Moving Average vs Date")
    return figure 

@app.callback(
    [Output('forecast_graph', 'figure'),
    Output('forecast-content', 'style')],
    [Input('forecast-val', 'n_clicks')],
    [State('number-days', 'value'),
    State('stock-code', 'value')]
)

def forecast_stock(n_clicks, days, v):
    if v==None:
        raise PreventUpdate
    
    if(n_clicks > 0):
        style = {'display':'block'}
    df = stock_predict(v, days)
    fig = get_predict(df, days)
    return [fig, style]

def get_predict(df, day):
    figure= px.line(df,
            x= 'Date', 
            y= 'Prediction',
            title="Predicted Close Price of next "+day+" days")
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
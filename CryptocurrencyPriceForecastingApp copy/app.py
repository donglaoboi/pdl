import plotly.graph_objects as go
import pathlib
import dash
import datetime
import numpy as np
import pandas as pd

from get_data import Dataset
from data_preprocessing import data_preprocessing
from dash import dcc, html, Input, Output
from binance.client import Client
from datetime import date, timedelta
from plotly.subplots import make_subplots

app = dash.Dash(
    __name__, meta_tags=[
        {"name": "viewport", "content": "width=device-width"}],
    update_title=None
)

app.title = "Cryptocurrency Price Forecasting"

server = app.server

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

currencies = ['BTCUSDT', 'BNBBUSD', 'ETHBUSD', 'SOLBUSD']

for currency in currencies:
    Dataset(interval='hour', coin=currency)
    # Dataset(interval='minute', coin=currency)

currency_pair_data = {}
for currency in currencies:
    currency_pair_data[f'{currency}_hour'] = data_preprocessing(
        'hour', currency).dataset
    # currency_pair_data[f'{currency}_minute'] = data_preprocessing(
    #     'minute', currency).dataset


def get_init_time():
    now = datetime.datetime.now()
    now = date(now.year, now.month, now.day)
    return dcc.DatePickerSingle(
        id='date-to-plot',
        min_date_allowed=date(2017, 8, 17),
        max_date_allowed=now,
        initial_visible_month=now,
        date=now
    )


app.layout = html.Div(
    className="row",
    children=[
        dcc.Interval(id="order_interval", interval=1 * 1000, n_intervals=0),
        dcc.Interval(id="trading_interval", interval=1 * 1000, n_intervals=0),
        dcc.Interval(id="clock_interval", interval=1 * 1000, n_intervals=0),
        dcc.Interval(id="data_interval", interval=1 * 3600000, n_intervals=0),
        dcc.Interval(id="graph_interval", interval=1 * 3600000, n_intervals=0),
        html.Div(id='hidden-div', style={'display': 'none'}),
        # Left Panel Div
        html.Div(
            className="three columns div-left-panel",
            children=[
                html.Div(
                    className="div-info",
                    children=[
                        html.H6(
                            children="Cryptocurrency Price Forecasting"
                        )
                    ],
                ),
                html.Div(
                    className="div-currency-toggles",
                    children=[
                        html.P(
                            id="live_clock",
                            className="three-col",
                            children=datetime.datetime.now().strftime("%H:%M:%S"),
                        ),
                        get_init_time(),
                        html.Div(
                            id="pairs",
                            className="div-bid-ask"
                        )
                    ],
                ),
            ],
        ),
        # Right Panel Div
        html.Div(
            className="nine columns div-right-panel",
            children=[
                html.Div(
                    className="row chart-top-bar",
                    children=[
                        dcc.Dropdown(
                            className="bottom-dropdown",
                            id="coin_selector",
                            options=[
                                {"label": "Bitcoin", "value": "BTCUSDT"},
                                {"label": "Binance Coin", "value": "BNBBUSD"},
                                {"label": "Ethereum Coin", "value": "ETHBUSD"},
                                {"label": "Solana Coin", "value": "SOLBUSD"},
                            ],
                            value="BTCUSDT",
                            clearable=False,
                        ),
                        dcc.Dropdown(
                            className="bottom-dropdown",
                            id="interval_selector",
                            options=[
                                # {"label": "15 minutes", "value": "minute"},
                                {"label": "1 hour", "value": "hour"}
                            ],
                            value="hour",
                            clearable=False,
                        )
                    ],
                ),
                html.Div(
                    id="charts",
                    className="row"
                ),
                html.Div(
                    id="bottom_panel",
                    className="row div-bottom-panel",
                    children=[
                        html.Div(
                            className="display-inlineblock",
                            children=[
                                dcc.Dropdown(
                                    id="dropdown_tradings",
                                    className="bottom-dropdown",
                                    options=[
                                        {
                                            "label": "5 Tradings",
                                            "value": 5
                                        },
                                        {
                                            "label": "10 Tradings",
                                            "value": 10
                                        },
                                        {
                                            "label": "15 Tradings",
                                            "value": 15
                                        },
                                    ],
                                    value=5,
                                    placeholder="Trading Limit",
                                    clearable=False,
                                    style={"border": "0px solid black"},
                                )
                            ],
                        ),
                        html.Div(id="trading_table",
                                 className="row table-orders"),
                    ],
                ),
            ],
        )
    ],
)


@app.callback(
    Output('hidden-div', 'children'),
    Input('data_interval', 'n_intervals')
)
def update_data(n):
    global currency_pair_data

    for currency in currencies:
        Dataset(interval='hour', coin=currency)
        # Dataset(interval='minute', coin=currency)

    currency_pair_data = {}
    for currency in currencies:
        currency_pair_data[f'{currency}_hour'] = data_preprocessing(
            'hour', currency).dataset
        # currency_pair_data[f'{currency}_minute'] = data_preprocessing(
        #     'minute', currency).dataset
    return None


@app.callback(
    Output('pairs', 'children'),
    [
        Input('order_interval', 'n_intervals'),
        Input('coin_selector', 'value')
    ]
)
def get_order(n, pair):
    client = Client()
    num_data = 10
    results = client.get_order_book(symbol=pair, limit=num_data)

    bids = results['bids']
    asks = results['asks']

    headers = [
        "Price",
        "Amount",
        "Total"
    ]

    bid_rows = []
    ask_rows = []

    for rows, df, color in zip([bid_rows, ask_rows], [bids, asks], ['#45df7e', '#da5657']):
        for row in df:
            tr_childs = []
            row[0] = round(float(row[0]), 2)
            row[1] = round(float(row[1]), 5)
            row.append(round(row[0] * row[1], 2))
            for i in range(3):
                tr_childs.append(
                    html.Td(
                        children=row[i],
                        style={'fontSize': '12px'}
                    )
                )
            rows.append(html.Tr(style={'color': color}, children=tr_childs))

    return html.Div(
        children=[
            html.Table(
                children=[
                    html.Tr(
                        [html.Th(
                            children=title,
                            style={'fontSize': '15px'}
                        ) for title in headers]
                    )
                ] + bid_rows),
            html.Table(
                children=[
                    html.Tr(
                        [html.Th(
                            children=title,
                            style={'fontSize': '15px'}
                        ) for title in headers]
                    )
                ] + ask_rows)
        ]
    )


@app.callback(
    Output('charts', 'children'),
    [
        Input('date-to-plot', 'date'),
        Input('coin_selector', 'value'),
        Input('interval_selector', 'value'),
        Input('graph_interval', 'n_intervals')
    ]
)
def kline_plot(date_value, pair, interval, n):
    data = currency_pair_data[f'{pair}_{interval}']

    date_value = date_value.split('-')
    year = int(date_value[0])
    month = int(date_value[1])
    day = int(date_value[2])

    if interval == 'hour':
        predicted_index = pd.date_range(start=datetime.datetime(
            year, month, day), periods=24, freq='H').to_pydatetime().tolist()
        previous_index = pd.date_range(end=datetime.datetime(
            year, month, day), periods=2, freq='H').to_pydatetime().tolist()[0]
    elif interval == 'minute':
        predicted_index = pd.date_range(start=datetime.datetime(
            year, month, day), periods=24 * 4, freq='15min').to_pydatetime().tolist()
        previous_index = pd.date_range(end=datetime.datetime(
            year, month, day), periods=2, freq='15min').to_pydatetime().tolist()[0]

    data_to_plot = data[(data['year'] == year) & (
        data['month'] == month) & (data['day'] == day)]

    threshold = len(data_to_plot)

    predicted_data = data[(data['year'] == previous_index.year) & (
        data['month'] == previous_index.month) & (data['day'] == previous_index.day)]

    predicted_data.index = predicted_index
    predicted_data_plot = predicted_data[threshold:]
    predicted_data_eval = predicted_data[:threshold]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        print_grid=False,
        vertical_spacing=0.12
    )

    colors = [['green', 'red'], ['white', 'black']]
    names = ['Real Time', 'Predicted']
    dfs = [data_to_plot, predicted_data_plot]

    for df, name, color in zip(dfs, names, colors):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                close=df['Close'],
                high=df['High'],
                low=df['Low'],
                increasing_line_color=color[0],
                decreasing_line_color=color[1],
                name=name
            ),
            row=1,
            col=1
        )

    dfs = [data_to_plot, predicted_data_eval]
    for df, name in zip(dfs, names):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name=name,
                mode='lines+markers'
            ),
            row=2,
            col=1
        )

    fig["layout"]["margin"] = {"t": 50, "l": 50, "b": 50, "r": 25}
    fig["layout"]["autosize"] = True
    fig['layout']['showlegend'] = False
    fig["layout"]["height"] = 400
    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
    fig["layout"]["xaxis"]["tickformat"] = "%H:%M"
    fig["layout"]["yaxis"]["showgrid"] = True
    fig["layout"]["yaxis"]["gridcolor"] = "#3E3F40"
    fig["layout"]["yaxis"]["gridwidth"] = 1
    fig["layout"]["xaxis2"]["rangeslider"]["visible"] = False
    fig["layout"]["xaxis2"]["tickformat"] = "%H:%M"
    fig["layout"]["yaxis2"]["showgrid"] = True
    fig["layout"]["yaxis2"]["gridcolor"] = "#3E3F40"
    fig["layout"]["yaxis2"]["gridwidth"] = 1
    fig["layout"].update(paper_bgcolor="#21252C", plot_bgcolor="#21252C")

    return html.Div(
        id=pair + "graph_div",
        children=[
            html.Div(
                dcc.Graph(
                    id=pair + "chart",
                    className="chart-graph",
                    config={"displayModeBar": False, "scrollZoom": True},
                    figure=fig
                )
            ),
        ],
    )


@app.callback(
    Output("live_clock", "children"),
    [Input("clock_interval", "n_intervals")]
)
def update_time(n):
    return datetime.datetime.now().strftime("%H:%M:%S")


@app.callback(
    Output("trading_table", "children"),
    [
        Input("dropdown_tradings", "value"),
        Input("trading_interval", "n_intervals"),
        Input('coin_selector', 'value')
    ]
)
def update_trading_table(limit, n, pair):
    headers = [
        "Time",
        "Price",
        "Quantity",
        "Quote Quantity"
    ]

    client = Client()
    trading_list = client.get_recent_trades(limit=limit, symbol=pair)

    rows = []
    item_list = ['time', 'price', 'qty', 'quoteQty']
    for trading in trading_list:
        tr_childs = []
        trading['time'] = datetime.datetime.fromtimestamp(
            trading['time'] / 1000).strftime("%H:%M:%S.%f")[:-4]
        for attr in item_list:
            tr_childs.append(html.Td(trading[attr]))
        if trading["isBuyerMaker"]:
            rows.append(
                html.Tr(style={'color': '#45df7e'}, children=tr_childs))
        else:
            rows.append(
                html.Tr(style={'color': '#da5657'}, children=tr_childs))

    return html.Table(children=[html.Tr([html.Th(title) for title in headers])] + rows)


if __name__ == '__main__':
    app.run_server(debug=True)

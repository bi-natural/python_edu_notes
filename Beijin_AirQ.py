"""
KDD CUP for Fresh Air: Beijing
"""
from math import sqrt
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM

import pandas as pd
import numpy as np
import os

#print("matplotlib backend = {}".format(matplotlib.rcParams['backend']))

# load data
#
# https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
#    -> https://archive.ics.uci.edu/ml/machine-learning-databases/00381/
#
""" Attribute Information:

:utctime: date format  -> YYYYMMDDHH0000
:stationid: station id (string)
:longitude: real
:latitude: real
:temperature: real    (5)
:pressure: real       (6)
:humidity: real       (7)
:winddirection: real  (8)
:windspeedkph: real   (9)
:weather: string
:PM25: real           (11)
:PM10: real           (12)
:NO2: real            (13)
:CO: real             (14)
:O3: real             (15)
:SO2: real            (16)

"""

def prepare_data(file='data/BeijingData_temp.csv'):
    if not os.path.exists(file):
        ValueError("file Not found {}".format(file))
        return

    # read data
    df = read_csv(file,  parse_dates=['utctime'], dayfirst=True)

    # generate new column 'weekday' from datetime
    df['weekday'] = df['utctime'].dt.dayofweek

    # reorder
    #     (stationid=1) + (utctime=0) + (weekday=-1) + ...
    cols = df.columns.tolist()
    df = df[[cols[1]] + [cols[0]] + [cols[-1]] + cols[2:-1]]

    # mark all NA values with each mean()
    df.replace(999017.0, np.nan, inplace=True)
    df.replace(999999.0, np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)

    # summarize first 5 rows
    # print(df.head(5))

    # save to file
    df.to_csv('data/KDD_Beijing.csv', index=False)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    :data: Sequence of observations as a list or NumPy array.
    :n_in: Number of lag observations as input (X).
    :n_out: Number of observations as output (y).
    :dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    :Pandas DataFrame of series framed for supervised learning.
    """
    # convert series to supervised learning
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def show_dataframe(df, head=5):
    if head > 0:
        print('------------- Head({}) --------------'.format(head))
        print(df.head(head))

    print('------------- DataFrame ---------------')
    print('temperature = {} ~ {}'.format(df['temperature'].min(), df['temperature'].max()))
    print('pressure    = {} ~ {}'.format(df['pressure'].min(), df['pressure'].max()))
    print('humidity    = {} ~ {}'.format(df['humidity'].min(), df['humidity'].max()))
    print('wind dir    = {} ~ {}'.format(df['winddirection'].min(), df['winddirection'].max()))
    print('wind speed  = {} ~ {}'.format(df['windspeedkph'].min(), df['windspeedkph'].max()))
    print('PM2.5       = {} ~ {}'.format(df['PM25'].min(), df['PM25'].max()))
    print('PM10        = {} ~ {}'.format(df['PM10'].min(), df['PM10'].max()))
    print('NO2         = {} ~ {}'.format(df['NO2'].min(), df['NO2'].max()))
    print('CO          = {} ~ {}'.format(df['CO'].min(), df['CO'].max()))
    print('O3          = {} ~ {}'.format(df['O3'].min(), df['O3'].max()))
    print('SO2         = {} ~ {}'.format(df['SO2'].min(), df['SO2'].max()))


def show_raw_data(file='data/KDD_Beijing.csv'):
    if not os.path.exists(file):
        ValueError("file Not found {}".format(file))
        return

    df = read_csv(file)

    # Show ranges of dataframe
    show_dataframe(df, 0)

    # get first station
    first_station = df['stationid'].iloc[0]
    df1 = df[df['stationid'] == first_station]

    # summarize first 5 rows of first stationid
    print(df1.head(5))

    # specify columns to plot (except = weather)
    groups = ['temperature', 'pressure', 'humidity', 'winddirection', 'windspeedkph']

    # plot each column
    i = 1
    pyplot.figure()
    for name in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(df1.ix[:, name])
        pyplot.title(name, y=0.5, loc='right')
        i += 1
    pyplot.savefig('img/input.png')


def load_data(file='data/KDD_Beijing.csv'):
    if not os.path.exists(file):
        ValueError("file Not found {}".format(file))
        return

    # load dataset
    df = read_csv(file)

    # show dataFrame shortly
    show_dataframe(df, 5)

    # get list of stations
    stations = df['stationid'].unique()
    print('get {} stations: {}'.format(len(stations), stations))

    data = [ ]
    for station in stations:
        df1 = df[df['stationid'] == station]
        df1 = df1.drop(columns=df1.columns[0], axis=1)

        # print('Shape = {}, Columns = {}'.format(df1.shape, df1.columns.tolist()))
        # integer encode weather (non integer/float type)
        encoder = LabelEncoder()
        df1.ix[:, 'utctime'] = encoder.fit_transform(df1.ix[:, 'utctime'])
        df1.ix[:, 'weather'] = encoder.fit_transform(df1.ix[:, 'weather'])

        # ensure all data is float
        values = df1.values
        values = values.astype('float32')

        print('station: {}, -> {}'.format(station, values[1, :]))

        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)

        # frame as supervised learning
        #print("before re-framed = shape {}".format(scaled.shape))
        #print(scaled[1, :])

        reframed = series_to_supervised(scaled, 1, 1)
        #print("after  re-framed = shape {}".format(reframed.shape))
        #print(reframed.head(1))

        # drop columns we don't want to predict
        reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
        #print("after  re-framed.drop = shape {}".format(reframed.shape))
        #print(reframed.head(1))

        item = {}
        item['station'] = station
        item['data'] = reframed
        item['scaler'] = scaler
        data.append(item)
    else:
        print('load {} stations data'.format(len(stations)))
        return data

def fetch_data(item):
    #
    # split into train and test sets
    #
    values = item['data'].values
    n_train_hours = values.shape[0] - 48
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    return item['station'], item['scaler'], train_X, train_y, test_X, test_y

def LSTM_Model(name, X_tr, y_tr):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_tr.shape[1], X_tr.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model


def LSTM_Run(name, model, X_tr, y_tr, X_te, y_te):
    history = model.fit(X_tr, y_tr, epochs=50, batch_size=72, validation_data=(X_te, y_te), verbose=2, shuffle=False)
    return history


def LSTM_Plot(name, history):
    pyplot.clf()
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.savefig('img/{}_hist.png'.format(name))


def LSTM_PredShow(name, inv_y, inv_yhat):
    pyplot.clf()
    pyplot.plot(inv_y, label='inv_y')
    pyplot.plot(inv_yhat, label='inv_yhat')
    pyplot.legend()
    pyplot.savefig('img/{}_test.png'.format(name))

    #print("inv_y")
    #print(inv_y.head(5))
    #print("inv_yhat")
    #print(inv_yhat.head(5))


def LSTM_Test(name, model, X_te, y_te, scaler):
    y_hat = model.predict(X_te)
    X_te = X_te.reshape((X_te.shape[0], X_te.shape[2]))

    # invert scaling for forecast
    inv_yhat = np.concatenate((y_hat, X_te[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    y_te = y_te.reshape((len(y_te), 1))
    inv_y = np.concatenate((y_te, X_te[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # Show data
    LSTM_PredShow(inv_y, inv_yhat)

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

if __name__ == '__main__':
    if not os.path.exists('data/KDD_Beijing.csv'):
        prepare_data()
        show_raw_data()

    # Train
    stations = load_data()

    for station in stations:
        name, scaler, X_tr, y_tr, X_te, y_te = fetch_data(station)
        print('-> {}'.format(name))
        model = LSTM_Model(name, X_tr, y_tr)
        history = LSTM_Run(name, model, X_tr, y_tr, X_te, y_te)
        LSTM_Plot(name, history)
        LSTM_Test(name, model, X_te, y_te, scaler)

    print('Done')

import pandas as pd
import numpy as np
import sklearn
import tensorflow
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from tensorflow import keras as keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from numpy import loadtxt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, concatenate, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV, KFold
tensorflow.keras.backend.set_floatx('float64')
from numpy import array

trnx = pd.read_csv("LOSDatatrnx.csv")
trny = pd.read_csv("LOSDatatrny.csv")
tstx = pd.read_csv("LOSDatatstx.csv")
tsty = pd.read_csv("LOSDatatsty.csv")

y = np.concatenate((trny, tsty))
x = np.concatenate((trnx,tstx))

trnx = trnx.to_numpy()
trny = trny.to_numpy()
tstx = tstx.to_numpy()
tsty = tsty.to_numpy()

trnx = trnx.reshape((trnx.shape[0], 336, 64))
tstx = tstx.reshape((tstx.shape[0], 336, 64))

x = x.reshape((x.shape[0], 336, 64))

kfold = KFold(n_splits=5)
kfold.get_n_splits(x)

cells = 128
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
cvscores = []
for train_index, test_index in kfold.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Sequential()
    model.add(LSTM(cells, return_sequences = True, input_shape=(trnx.shape[1], trnx.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(cells, return_sequences = True))
    model.add(Dropout(0.3))
    model.add(LSTM(cells, return_sequences = True))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(1, activation = 'relu')))
    model.compile(optimizer= opt , loss= tensorflow.keras.losses.MeanSquaredLogarithmicError() , 
                    metrics=['accuracy', 'mean_absolute_error', 'mse' ])

    #MAD, MAPE, MSE, MSLE, R2, Kappa 
    history = model.fit(trnx, trny, 
                    epochs=8, batch_size = 512, validation_split=0.1, verbose=2)

    print("Evaluate on test data")
    results = model.evaluate(tstx, tsty, batch_size = 512)
    print("MSLE, ACC, MAE, MSE:", results)
    print("%s: %.2f%%" % (model.metrics_names[1], results[1]*100))
    cvscores.append(results[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

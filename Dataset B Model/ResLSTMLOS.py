import pandas as pd
import numpy as np
import sklearn
import tensorflow
from sklearn import preprocessing
from sklearn import model_selection
from tensorflow import keras as keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from numpy import loadtxt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, Add,Dropout,Bidirectional, TimeDistributed
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV, KFold
tensorflow.keras.backend.set_floatx('float64')
from numpy import array

trnx = pd.read_csv("/cs/home/psxau1/db/HalfLOSDatatrnx.csv")
trny = pd.read_csv("/cs/home/psxau1/db/HalfLOSDatatrny.csv")
tstx = pd.read_csv("/cs/home/psxau1/db/HalfLOSDatatstx.csv")
tsty = pd.read_csv("/cs/home/psxau1/db/HalfLOSDatatsty.csv")

trnx = trnx.to_numpy()
trny = trny.to_numpy()
tstx = tstx.to_numpy()
tsty = tsty.to_numpy()

trnx = trnx.reshape((trnx.shape[0], 336, 35))
tstx = tstx.reshape((tstx.shape[0], 336, 35))

y = np.concatenate((trny, tsty))
x = np.concatenate((trnx,tstx))

x = x.reshape((x.shape[0], 336, 35))

kfold = KFold(n_splits=5)
kfold.get_n_splits(x)

cvscores = []

cells = 128
epochs = 15
batch_size = 64
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)

for train_index, test_index in kfold.split(x):
    x_train, x_test = x[train_index, :, :], x[test_index, :, :]
    y_train, y_test = y[train_index], y[test_index]
    input_shape = Input(shape=(x_train.shape[1], x_train.shape[2]))

    layerone = LSTM(cells, return_sequences = True) (input_shape)
    layerone = Dropout(0.3)(layerone)
    layertwo = LSTM(cells, return_sequences = True) (layerone)
    layertwo = Dropout(0.3)(layertwo)
    mergetwo = Add()([layerone, layertwo])
    layerthree = LSTM(cells, return_sequences = True) (mergetwo)
    layerthree = Dropout(0.3)(layerthree)
    mergethree = Add()([layerone, layertwo, layerthree])
    layerfour = LSTM(cells, return_sequences = True) (mergethree)
    layerfour = Dropout(0.3)(layerfour)
    mergefour = Add()([layerone, layertwo, layerthree, layerfour])
    layerfive = LSTM(cells, return_sequences = True) (mergefour)
    layerfive = Dropout(0.3)(layerfive)
    mergefive = Add()([layerone, layertwo, layerthree, layerfour, layerfive])
    layersix = LSTM(cells, return_sequences = True) (mergefive)
    layersix = Dropout(0.3)(layersix)
    mergesix = Add()([layerone, layertwo, layerthree, layerfour, layerfive, layersix])
    layerseven = LSTM(cells, return_sequences = True) (mergesix)
    layerseven = Dropout(0.3)(layerseven)
    
    mergeseven = Add()([layerone, layertwo, layerthree, layerfour, layerfive, layersix, layerseven])
    layereight = LSTM(cells, return_sequences = True) (mergeseven)
    layereight = Dropout(0.3)(layereight)
    mergeeight = Add()([layerone, layertwo, layerthree, layerfour, layerfive, layersix, layerseven, layereight])
    layernine = LSTM(cells, return_sequences = True) (mergeeight)
    layernine = Dropout(0.3)(layernine)
    mergenine = Add()([layerone, layertwo, layerthree, layerfour, layerfive, layersix, layerseven, layereight, layernine])
    layerten = LSTM(cells, return_sequences = True) (mergenine)
    layerten = Dropout(0.3)(layerten)
    out = TimeDistributed(Dense(1, activation='relu'))(layerten)
    model = Model(inputs=input_shape, outputs=out)
    model.compile(optimizer= opt , loss= "mean_squared_logarithmic_error" , 
                metrics=['mean_absolute_error', 'mse' ])
    history = model.fit(x_train, y_train,epochs=epochs, batch_size = batch_size, verbose=2)
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size = batch_size)
    print("MSLE, MAE, MSE:", results)
    print("%s: %.2f%%" % (model.metrics_names[0], results[0]))
    cvscores.append(results[0])
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

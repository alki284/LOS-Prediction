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
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, Add,Dropout,Bidirectional, TimeDistributed
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV, KFold
tensorflow.keras.backend.set_floatx('float64')
from numpy import array

trnx = pd.read_csv("trnxSAPSnumpy.csv")
trny = pd.read_csv("trnySAPSnumpy.csv")
tstx = pd.read_csv("tstxSAPSnumpy.csv")
tsty = pd.read_csv("tstySAPSnumpy.csv")

trnx = trnx.to_numpy()
trny = trny.to_numpy()
tstx = tstx.to_numpy()
tsty = tsty.to_numpy()

trnx = trnx.reshape((trnx.shape[0], 336, 35))
tstx = tstx.reshape((tstx.shape[0], 336, 35))

cells = 128
epochs = 1
batch_size = 64
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
input_shape = Input(shape=(trnx.shape[1], trnx.shape[2]))

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
out = TimeDistributed(Dense(1, activation='relu'))(layersix)
model = Model(inputs=input_shape, outputs=out)
model.compile(optimizer= opt , loss= "mean_squared_logarithmic_error" , 
            metrics=['mean_absolute_error', 'mse' ])
history = model.fit(trnx, trny,epochs=epochs, batch_size = batch_size, verbose=2)
print("Evaluate on test data")
results = model.evaluate(tstx, tsty, batch_size = batch_size)
print("MSLE, ACC, MAE, MSE:", results)

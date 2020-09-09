import numpy as np
import pandas as pd
import sklearn
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import make_column_transformer
from numpy import loadtxt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, concatenate, Dropout, TimeDistributed, Add
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow.keras.metrics
from numpy import array

trnx = pd.read_csv("trnxSAPSnumpy.csv")
trny = pd.read_csv("trnySAPSnumpy.csv")
tstx = pd.read_csv("tstxSAPSnumpy.csv")
tsty = pd.read_csv("tstySAPSnumpy.csv")

y = np.concatenate((trny, tsty))
x = np.concatenate((trnx,tstx))

x = x.reshape((x.shape[0], 336, 35))

kfold = KFold(n_splits=5)
kfold.get_n_splits(x)

cvscores = []



for train_index, test_index in kfold.split(x):
    x_train, x_test = x[train_index, :, :], x[test_index, :, :]
    y_train, y_test = y[train_index], y[test_index]

    HR = x_train[:,:,[0,1]]
    BP = x_train[:,:,[2,3]]
    Pot = x_train[:,:,[4,5]]
    Temp = x_train[:,:,[6,7]]
    WBC = x_train[:,:,[8,9]]
    Na = x_train[:,:,[10,11]]
    Bi = x_train[:,:,[12,13]]
    Coma = x_train[:,:,[14,15]]
    Biri = x_train[:,:,[16,17]]
    BUN = x_train[:,:,[18,19]]
    Urine = x_train[:,:,[20,21]]
    Vent = x_train[:,:,[22,23]]
    Admit = x_train[:,:,[24, 25]]
    Age = x_train[:, :,[26, 27, 28, 29, 30, 31]]
    infect = x_train[:,:,[32]]
    Cancer = x_train[:,:,[33]]
    Blood = x_train[:,:,[34]]

    HRtst = x_test[:,:,[0,1]]
    BPtst = x_test[:,:,[2,3]]
    Pottst = x_test[:,:,[4,5]]
    Temptst = x_test[:,:,[6,7]]
    WBCtst = x_test[:,:,[8,9]]
    Natst = x_test[:,:,[10,11]]
    Bitst = x_test[:,:,[12,13]]
    Comatst = x_test[:,:,[14,15]]
    Biritst = x_test[:,:,[16,17]]
    BUNtst = x_test[:,:,[18,19]]
    Urinetst = x_test[:,:,[20,21]]
    Venttst = x_test[:,:,[22,23]]
    Admittst = x_test[:,:,[24, 25]]
    Agetst = x_test[:, :,[26, 27, 28, 29, 30, 31]]
    infecttst = x_test[:,:,[32]]
    Cancertst = x_test[:,:,[33]]
    Bloodtst = x_test[:,:,[34]]

    inpHR = Input(shape=(HR.shape[1], HR.shape[2]))
    inpBP = Input(shape=(BP.shape[1], BP.shape[2]))
    inpPot = Input(shape=(Pot.shape[1], Pot.shape[2]))
    inpTemp = Input(shape=(Temp.shape[1], Temp.shape[2]))
    inpWBC = Input(shape=(WBC.shape[1], WBC.shape[2]))
    inpNa = Input(shape=(Na.shape[1], Na.shape[2]))
    inpBi = Input(shape=(Bi.shape[1], Bi.shape[2]))
    inpComa = Input(shape=(Coma.shape[1], Coma.shape[2]))
    inpBiri = Input(shape=(Biri.shape[1], Biri.shape[2]))
    inpBUN = Input(shape=(BUN.shape[1], BUN.shape[2]))
    inpUrine = Input(shape=(Urine.shape[1], Urine.shape[2]))
    inpVent = Input(shape=(Vent.shape[1], Vent.shape[2]))
    inpAdmit = Input(shape=(Admit.shape[1], Admit.shape[2]))
    inpAge = Input(shape=(Age.shape[1], Age.shape[2]))
    inpinfect = Input(shape=(infect.shape[1], infect.shape[2]))
    inpCancer = Input(shape=(Cancer.shape[1], Cancer.shape[2]))
    inpBlood = Input(shape=(Blood.shape[1], Blood.shape[2]))

    x_train = [HR, BP, Pot, Temp, WBC, Na, Bi, Coma, Biri, BUN, Urine, Vent, Admit, Age, infect,
                    Cancer, Blood]
    x_test = [HRtst, BPtst, Pottst, Temptst, WBCtst, Natst, Bitst, Comatst,Biritst, BUNtst, Urinetst, Venttst,
   Admittst, Agetst, infecttst, Cancertst, Bloodtst]
    cells = 128
    chancells = 64
    epochs = 25
    learning_rate = 0.01
    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
    dropout_rate = 0.3
    batch_size = 64

    HR_input = LSTM(chancells, return_sequences=True)(inpHR)
    BP_input = LSTM(chancells,  return_sequences=True)(inpBP)
    Pot_input = LSTM(chancells,  return_sequences=True)(inpPot)
    Temp_input = LSTM(chancells,  return_sequences=True)(inpTemp)
    WBC_input = LSTM(chancells, return_sequences=True)(inpWBC)
    Na_input = LSTM(chancells, return_sequences=True)(inpNa)
    Bi_input = LSTM(chancells,  return_sequences=True)(inpBi)
    Coma_input = LSTM(chancells,  return_sequences=True)(inpComa)
    Biri_input = LSTM(chancells,  return_sequences=True)(inpBiri)
    BUN_input = LSTM(chancells,  return_sequences=True)(inpBUN)
    Urine_input = LSTM(chancells,  return_sequences=True)(inpUrine)
    Vent_input = LSTM(chancells,  return_sequences=True)(inpVent)
    Admit_input = LSTM(chancells,  return_sequences=True)(inpAdmit)
    Age_input = LSTM(chancells,  return_sequences=True)(inpAge)
    infect_input = LSTM(chancells,  return_sequences=True)(inpinfect)
    Cancer_input = LSTM(chancells,  return_sequences=True)(inpCancer)
    Blood_input = LSTM(chancells,  return_sequences=True)(inpBlood)

    merge_one = concatenate([HR_input, BP_input, Pot_input, Temp_input, WBC_input, Na_input, Bi_input, Coma_input,
                        Biri_input, BUN_input, Urine_input, Vent_input, Admit_input, Age_input, infect_input, 
                        Cancer_input, Blood_input])

    layerone = LSTM(cells, return_sequences = True) (merge_one)
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
    out =  TimeDistributed(Dense(1, activation='relu'))(layersix)
    model = Model(inputs=[inpHR, inpBP, inpPot, inpTemp, inpWBC, inpNa, inpBi, inpComa, inpBiri, inpBUN,
                    inpUrine, inpVent, inpAdmit, inpAge, inpinfect, inpCancer, inpBlood], 
                  outputs=out)
    model.compile(optimizer= opt , loss= "mean_squared_logarithmic_error" , 
            metrics=['mean_absolute_error', 'mse' ])

    
    history = model.fit(x_train, y_train,epochs=epochs, batch_size = batch_size, verbose=2)
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size = batch_size)
    print("MSLE, MAE, MSE:", results)
    print("%s: %.2f%%" % (model.metrics_names[0], results[0]))
    cvscores.append(results[0])
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    

    

    





    
    

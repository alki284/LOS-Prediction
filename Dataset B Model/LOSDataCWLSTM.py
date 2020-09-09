import numpy as np
import pandas as pd
import sklearn
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import make_column_transformer
from numpy import loadtxt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, concatenate, Dropout, TimeDistributed
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow.keras.metrics
from numpy import array

trnx = pd.read_csv("/cs/home/psxau1/db/LOSDatatrnx.csv")
trny = pd.read_csv("/cs/home/psxau1/db/LOSDatatrny.csv")
tstx = pd.read_csv("/cs/home/psxau1/db/LOSDatatstx.csv")
tsty = pd.read_csv("/cs/home/psxau1/db/LOSDatatsty.csv")

y = np.concatenate((trny, tsty))
x = np.concatenate((trnx,tstx))

x = x.reshape((x.shape[0], 336, 64))

kfold = KFold(n_splits=5)
kfold.get_n_splits(x)

cvscores = []

for train_index, test_index in kfold.split(x):
    x_train, x_test = x[train_index, :, :], x[test_index, :, :]
    y_train, y_test = y[train_index], y[test_index]

    HR = x_train[:,:,[0,1]]
    Temp = x_train[:,:,[2,3]]
    WBC = x_train[:,:,[4,5]]
    Glas = x_train[:,:,[6,7]]
    Mech_Vent = x_train[:,:,[8,9]]
    Biril = x_train[:,:,[10,11]]
    BUN = x_train[:,:,[12,13]]
    Albumin = x_train[:,:,[14,15]]
    OxySat = x_train[:,:,[16,17]]
    Platelets = x_train[:,:,[18,19]]
    RBC = x_train[:,:,[20,21]]
    pH = x_train[:,:,[22,23]]
    Hematocrit = x_train[:,:,[24, 25]]
    ResRate = x_train[:, :,[26, 27]]
    Weight = x_train[:, :, [28, 29]]
    HADM = x_train[:, :, [30]]
    Blood = x_train[:, :, [31]]
    Digestive = x_train[:, :, [32]]
    Genitourinary = x_train[:, :, [33]]
    Illdefined = x_train[:,:,[34]]
    Infectious = x_train[:, :, [35]]
    Injury = x_train[:, :, [36]]
    Nervous = x_train[:, :, [37]]
    Respiratory = x_train[:, :, [38]]
    Skin = x_train[:, :, [39]]
    Marriage = x_train[:, :, [40, 41, 42, 43]]
    Religion = x_train[:, :, [44, 45, 46, 47, 48, 49]]
    Insurance = x_train[:, :, [50, 51, 52]]
    AdmissionArea = x_train[:, :, [53, 54, 55, 56, 57, 58, 59, 60]]
    AdmissionType = x_train[:, :, [61, 62, 63]]


    HRtst = x_test[:,:,[0,1]]
    Temptst = x_test[:,:,[2,3]]
    WBCtst = x_test[:,:,[4,5]]
    Glastst = x_test[:,:,[6,7]]
    Mech_Venttst = x_test[:,:,[8,9]]
    Biriltst = x_test[:,:,[10,11]]
    BUNtst = x_test[:,:,[12,13]]
    Albumintst = x_test[:,:,[14,15]]
    OxySattst = x_test[:,:,[16,17]]
    Plateletstst = x_test[:,:,[18,19]]
    RBCtst = x_test[:,:,[20,21]]
    pHtst = x_test[:,:,[22,23]]
    Hematocrittst = x_test[:,:,[24, 25]]
    ResRatetst = x_test[:, :,[26, 27]]
    Weighttst = x_test[:, :, [28, 29]]
    HADMtst = x_test[:, :, [30]]
    Bloodtst = x_test[:, :, [31]]
    Digestivetst = x_test[:, :, [32]]
    Genitourinarytst = x_test[:, :, [33]]
    Illdefinedtst = x_test[:,:,[34]]
    Infectioustst = x_test[:, :, [35]]
    Injurytst = x_test[:, :, [36]]
    Nervoustst = x_test[:, :, [37]]
    Respiratorytst = x_test[:, :, [38]]
    Skintst = x_test[:, :, [39]]
    Marriagetst = x_test[:, :, [40, 41, 42, 43]]
    Religiontst = x_test[:, :, [44, 45, 46, 47, 48, 49]]
    Insurancetst = x_test[:, :, [50, 51, 52]]
    AdmissionAreatst = x_test[:, :, [53, 54, 55, 56, 57, 58, 59, 60]]
    AdmissionTypetst = x_test[:, :, [61, 62, 63]]

    

    inpHR = Input(shape=(HR.shape[1], HR.shape[2]))
    inpTemp = Input(shape=(Temp.shape[1], Temp.shape[2]))
    inpWBC = Input(shape=(WBC.shape[1], WBC.shape[2]))
    inpGlas = Input(shape=(Glas.shape[1], Glas.shape[2]))
    inpMech_Vent = Input(shape=(Mech_Vent.shape[1], Mech_Vent.shape[2]))
    inpBiril = Input(shape=(Biril.shape[1], Biril.shape[2]))
    inpBUN = Input(shape=(BUN.shape[1], BUN.shape[2]))
    inpAlbumin = Input(shape=(Albumin.shape[1], Albumin.shape[2]))
    inpOxySat = Input(shape=(OxySat.shape[1], OxySat.shape[2]))
    inpPlatelets = Input(shape=(Platelets.shape[1], Platelets.shape[2]))
    inpRBC = Input(shape=(RBC.shape[1], RBC.shape[2]))
    inppH = Input(shape=(pH.shape[1], pH.shape[2]))
    inpHematocrit = Input(shape=(Hematocrit.shape[1], Hematocrit.shape[2]))
    inpResRate = Input(shape=(ResRate.shape[1], ResRate.shape[2]))
    inpWeight = Input(shape=(Weight.shape[1], Weight.shape[2]))
    inpHADM = Input(shape=(HADM.shape[1], HADM.shape[2]))
    inpBlood = Input(shape=(Blood.shape[1], Blood.shape[2]))
    inpDigestive = Input(shape=(Digestive.shape[1], Digestive.shape[2]))
    inpGenitourinary = Input(shape=(Genitourinary.shape[1], Genitourinary.shape[2]))
    inpIlldefined = Input(shape=(Illdefined.shape[1], Illdefined.shape[2]))
    inpInfectious = Input(shape=(Infectious.shape[1], Infectious.shape[2]))
    inpInjury = Input(shape=(Injury.shape[1], Injury.shape[2]))
    inpNervous = Input(shape=(Nervous.shape[1], Nervous.shape[2]))
    inpRespiratory = Input(shape=(Respiratory.shape[1], Respiratory.shape[2]))
    inpSkin = Input(shape=(Skin.shape[1], Skin.shape[2]))
    inpMarriage = Input(shape=(Marriage.shape[1], Marriage.shape[2]))
    inpReligion = Input(shape=(Religion.shape[1], Religion.shape[2]))
    inpInsurance = Input(shape=(Insurance.shape[1], Insurance.shape[2]))
    inpAdmissionArea = Input(shape=(AdmissionArea.shape[1], AdmissionArea.shape[2]))
    inpAdmissionType = Input(shape=(AdmissionType.shape[1], AdmissionType.shape[2]))

    x_train = [HR,Temp, WBC, Glas, Mech_Vent, Biril, BUN, Albumin, OxySat, Platelets, RBC, pH, Hematocrit, ResRate, Weight, HADM,
               Blood, Digestive, Genitourinary, Illdefined, Infectious, Injury, Nervous, Respiratory, Skin, Marriage, Religion,
               Insurance, AdmissionArea, AdmissionType]

    x_test = [HRtst,Temptst, WBCtst, Glastst, Mech_Venttst, Biriltst, BUNtst, Albumintst, OxySattst, Plateletstst, RBCtst, pHtst,
              Hematocrittst, ResRatetst, Weighttst, HADMtst, Bloodtst, Digestivetst, Genitourinarytst, Illdefinedtst, Infectioustst,
              Injurytst, Nervoustst, Respiratorytst, Skintst, Marriagetst, Religiontst,
               Insurancetst, AdmissionAreatst, AdmissionTypetst]
    
    cells = 128
    chancells = 64
    epochs = 12
    learning_rate = 0.01
    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
    dropout_rate = 0.3
    batch_size = 128

    HR_input = LSTM(chancells, return_sequences=True)(inpHR)
    Temp_input = LSTM(chancells, return_sequences=True)(inpTemp)
    WBC_input = LSTM(chancells, return_sequences=True)(inpWBC)
    Glas_input = LSTM(chancells, return_sequences=True)(inpGlas)
    Mech_Vent_input = LSTM(chancells, return_sequences=True)(inpMech_Vent)
    Biril_input = LSTM(chancells, return_sequences=True)(inpBiril)
    BUN_input = LSTM(chancells, return_sequences=True)(inpBUN)
    Albumin_input = LSTM(chancells, return_sequences=True)(inpAlbumin)
    OxySat_input = LSTM(chancells, return_sequences=True)(inpOxySat)
    Platelets_input = LSTM(chancells, return_sequences=True)(inpPlatelets)
    RBC_input = LSTM(chancells, return_sequences=True)(inpRBC)
    pH_input = LSTM(chancells, return_sequences=True)(inppH)
    Hematocrit_input = LSTM(chancells, return_sequences=True)(inpHematocrit)
    ResRate_input = LSTM(chancells, return_sequences=True)(inpResRate)
    Weight_input = LSTM(chancells, return_sequences=True)(inpWeight)
    HADM_input = LSTM(chancells, return_sequences=True)(inpHADM)
    Blood_input = LSTM(chancells, return_sequences=True)(inpBlood)
    Digestive_input = LSTM(chancells, return_sequences=True)(inpDigestive)
    Genitourinary_input = LSTM(chancells, return_sequences=True)(inpGenitourinary)
    Illdefined_input = LSTM(chancells, return_sequences=True)(inpIlldefined)
    Infectious_input = LSTM(chancells, return_sequences=True)(inpInfectious)
    Injury_input = LSTM(chancells, return_sequences=True)(inpInjury)
    Nervous_input = LSTM(chancells, return_sequences=True)(inpNervous)
    Respiratory_input = LSTM(chancells, return_sequences=True)(inpRespiratory)
    Skin_input = LSTM(chancells, return_sequences=True)(inpSkin)
    Marriage_input = LSTM(chancells, return_sequences=True)(inpMarriage)
    Religion_input = LSTM(chancells, return_sequences=True)(inpReligion)
    Insurance_input = LSTM(chancells, return_sequences=True)(inpInsurance)
    AdmissionArea_input = LSTM(chancells, return_sequences=True)(inpAdmissionArea)
    AdmissionType_input = LSTM(chancells, return_sequences=True)(inpAdmissionType)


    merge_one = concatenate([HR_input, Temp_input, WBC_input, Glas_input, Mech_Vent_input, Biril_input, BUN_input, Albumin_input,
                            OxySat_input, Platelets_input, RBC_input, pH_input, Hematocrit_input, ResRate_input, Weight_input,
                             HADM_input, Blood_input, Digestive_input, Genitourinary_input, Illdefined_input, Infectious_input,
                             Injury_input, Nervous_input, Respiratory_input, Skin_input, Marriage_input, Religion_input,
                             Insurance_input, AdmissionArea_input, AdmissionType_input])

    first = LSTM(cells,  return_sequences=True)(merge_one)
    first = Dropout(dropout_rate)(first)
    final = LSTM(cells,  return_sequences=True)(first)
    final = Dropout(dropout_rate)(final)
    out =  TimeDistributed(Dense(1, activation='relu'))(final)
    model = Model(inputs=[inpHR,inpTemp, inpWBC, inpGlas, inpMech_Vent, inpBiril, inpBUN, inpAlbumin, inpOxySat, inpPlatelets, inpRBC, inppH,
                          inpHematocrit, inpResRate, inpWeight, inpHADM,
               inpBlood, inpDigestive, inpGenitourinary, inpIlldefined, inpInfectious, inpInjury, inpNervous, inpRespiratory, inpSkin, inpMarriage,
                          inpReligion, inpInsurance, inpAdmissionArea, inpAdmissionType], 
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

    

    

    





    
    

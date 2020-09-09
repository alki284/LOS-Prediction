import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import time
import statistics
%matplotlib inline
from datetime import date
import scipy
from scipy.stats import ttest_ind

df_time = pd.read_csv("50varraw.csv")

df_time = pd.DataFrame(df_time)

df_time = df_time.drop_duplicates(subset='HADM_ID', keep='first')

df_time = df_time.reindex()

df_ICD = df_time[['blood and blood-forming organs', 'circulatory', 'congenital',
       'digestive', 'endocrine, metabolic and immunity',
       'external causes of injury and supplemental classification',
       'genitourinary', 'ill-defined', 'infectious and parasitic',
       'injury and poisoning', 'mental', 'musculoskeletal', 'neoplasms',
       'nervous', 'perinatal period', 'pregnancy, childbirth, and puerperium',
       'respiratory', 'skin and subcutaneous tissue', 'LOS_hr']]

# Function finds ttest result for an ICD9 block

def ttest(index, name):
    one = []
    zero = []
    for i in range(0, 22576):
        if df_ICD.iloc[i,index] == 1:
            one.append(df_ICD.iloc[i, -1])
        else:
            zero.append(df_ICD.iloc[i, -1])
    print(ttest_ind(one,zero, equal_var = False))

ttest(0, 'blood and blood-forming organs')
ttest(1, 'circulatory')
ttest(2, 'congenital')
ttest(3, 'digestive')
ttest(4, 'endocrine, metabolic and immunity')
ttest(5, 'external causes of injury and supplemental classification')
ttest(6, 'genitourinary')
ttest(7, 'ill-defined')
ttest(8, 'infectious and parasitic')
ttest(9, 'injury and poisoning')
ttest(10, 'mental')
ttest(11, 'musculoskeletal')
ttest(12, 'neoplasms')
ttest(13,  'nervous')
ttest(14, 'perinatal period')
ttest(15, 'pregnancy, childbirth, and puerperium')
ttest(16, 'respiratory')
ttest(17, 'skin and subcutaneous tissue')

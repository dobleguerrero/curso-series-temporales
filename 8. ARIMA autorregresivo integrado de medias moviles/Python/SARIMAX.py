import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings("ignore")

archivo = "Index2018.csv"
rawData = pd.read_csv(archivo)

dfTrain = rawData.copy()
dfTrain.date = pd.to_datetime(dfTrain.date, dayfirst=True)
dfTrain.set_index("date", inplace=True)
# h: hourly, w: weekly, d: daily, m: monthly, a: anual, b: business days
df = dfTrain.asfreq("b")
# ffill: front filling, asigna el valor del periodo posterior
# bfill: back filling, asigna el valor del periodo anterior
# asignar valor constante
# para series de tiempo no aplica en la mayoria de los casos
df = dfTrain.fillna(method="ffill")
dfTrain["marketValue"] = dfTrain.ftse
# eliminar columnas
# del dfTrain["spx"]
# del dfTrain["dax"]
# del dfTrain["ftse"]
# del dfTrain["nikkei"]

size = int(len(df) * 0.8)
dfTrain, dfTest = dfTrain.iloc[:size], dfTrain.iloc[size:]

model_sarimax = SARIMAX(dfTrain.marketValue, exog = dfTrain.spx, order=(1,0,1), seasonal_order = (2,0,1,5))
results_sarimax = model_sarimax.fit()
results_sarimax.summary()


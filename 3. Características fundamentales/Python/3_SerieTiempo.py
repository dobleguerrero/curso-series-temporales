import pandas as pd
import numpy as np

archivo = "Index2018.csv"
rawData = pd.read_csv(archivo)

df = rawData.copy()
df.date = pd.to_datetime(df.date, dayfirst=True)
df.set_index("date", inplace=True)
# h: hourly, w: weekly, d: daily, m: monthly, a: anual, b: business days
df = df.asfreq("b")
# ffill: front filling, asigna el valor del periodo posterior
# bfill: back filling, asigna el valor del periodo anterior
# asignar valor constante
# para series de tiempo no aplica en la mayoria de los casos
df = df.fillna(method="ffill")
df["marketValue"] = df.spx
# eliminar columnas
del df["spx"]
del df["dax"]
del df["ftse"]
del df["nikkei"]

size = int(len(df) * 0.8)
dfTrain, dfTest = df.iloc[:size], df.iloc[size:]

## RUIDO BLANCO
import matplotlib.pyplot as plt
wn = np.random.normal(loc=dfTrain.marketValue.mean(), scale=dfTrain.marketValue.std(), size=len(dfTrain))
dfTrain["wn"] = wn

archivo = "RandWalk.csv"
rw = pd.read_csv(archivo)
rw.date = pd.to_datetime(rw.date, dayfirst=True)
rw.set_index("date", inplace=True)
# h: hourly, w: weekly, d: daily, m: monthly, a: anual, b: business days
rw = rw.asfreq("b")

#print(df.isna().sum())

dfTrain["rw"] = rw.price

print(dfTrain)

# dfTrain.rw.plot(figsize=(10, 5))
# dfTrain.marketValue.plot(figsize=(10,5))
# plt.title("Random Walk vs SPX Prices", size=24)
# plt.legend()
# plt.show()

## ESTACIONARIEDAD
# procesos estacionarios
# hipotesis nula: La seria no es estacionaria
# hipotesis alternativo: La serie es estacionaria

# contraste de hipotesis
# estadistico de contraste
# si estadistico de contraste > valor critico -> aceptamos la hipotesis nula
# si estadistico de contraste < valor critico -> rechazamos la hipotesis nula
# p valor
# si p-valor > nivel de significancia -> aceptamos la hipotesis nula
# si p-valor < nivel de significancia -> rechazamos la hipotesis nula

import statsmodels.tsa.stattools as sts

print(sts.adfuller(dfTrain.marketValue))
print(sts.adfuller(dfTrain.wn))
print(sts.adfuller(dfTrain.rw))

# estadistico de contraste
# p-value
# numero de retrasos utilizados (series estacionarias regularmente  no son mayor que 0)
# {nivel de significancia: valor critico}

## ESTACIONALIDAD
# patrones ciclos
# descomposicion de una serie de tiempo
# Tendencia: presencia de patron consistente
# Estacional: efectos ciclicos debido a la estacionalidad
# Residual: error de predicción entre predicciones y valor real

# descomposicion clasica
# aditivo: para cualquier periodo de tiempo el valor observado es
#          la suma de la tendencia, el efecto estacional y el efecto residual
# multiplicativo: para cualquier periodo de tiempo el valor observado es
#                 el producto de la tendencia, el efecto estacional y el efecto residual

from statsmodels.tsa.seasonal import seasonal_decompose
additive = seasonal_decompose(dfTrain.marketValue, model="additive")
additive.plot()
plt.show()

multiplicative = seasonal_decompose(dfTrain.marketValue, model="multiplicative")
multiplicative.plot()
plt.show()

#
# import statsmodels.graphics.tsaplots as sgt
# import seaborn as sns
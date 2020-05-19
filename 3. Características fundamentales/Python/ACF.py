import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt

# AUTOCORRELACION
# Correlacion entre una secuencia y si misma
# Nivel de semejanza entre una secuencia en diferentes periodos

# ACF: AutoCorrelation Function
# Autocorrelacion para cualquier retraso a considerar

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

# Grafica de autocorrelacion
# la practica comun es analizar los primeros 40 retrasos
# zero=False omite la autocorrelacion consigo mismo

# Se observa que todas las lineas son mas altas que la zona sombreada en azul
# Esto sugiere que los coeficientes son significativos
# por lo que es un indicador de la dependencia del tiempo en los datos
# existe autocorrelacion
sgt.plot_acf(dfTrain.marketValue, lags=40, zero=False)
plt.title("ACF SPX", size=24)
# plt.show()

# Se observa que practicamente todas las lineas estan dentro la zona sombreada en azul
# Esto sugiere que los coeficientes no son significativos
# por lo que es un indicador de que no existe una dependencia del tiempo en los datos
# no existe autocorrelacion
sgt.plot_acf(dfTrain.wn, lags=40, zero=False)
plt.title("ACF WN", size=24)
# plt.show()

# Se observa que todas las lineas son mas altas que la zona sombreada en azul
# Esto sugiere que los coeficientes son significativos
# por lo que es un indicador de la dependencia del tiempo en los datos
# existe autocorrelacion
sgt.plot_acf(dfTrain.rw, lags=40, zero=False)
plt.title("ACF RW", size=24)
# plt.show()

# AUTOCORRELACION PARCIAL
# PACF: Parcial AutoCorrelation Function
# Ignora la autocorrelacion indirecta, es decir,
# la afectacion intermedias que se obtiene a traves de otros canales a los datos actuales
# Por ejemplo, en una serie diaria, el dia 1, afecta al dia 2, 3, 4
# pero a su vez el dia 2 afecta al dia 3, 4 y el dia 3 al dia 4
# en la autocorrelacion parcial se calcula solo la correlacion directa
# es decir, como afecta el dia 1 al dia 4, el dia 2 al dia 4, el dia 3 al dia 4

# Se observa que todas las lineas son mas altas que la zona sombreada en azul
# Esto sugiere que los coeficientes son significativos
# por lo que es un indicador de la dependencia del tiempo en los datos
# existe autocorrelacion
sgt.plot_pacf(dfTrain.marketValue, lags=40, zero=False, method="ols")
plt.title("PACF SPX", size=24)
# plt.show()

# Se observa que practicamente todas las lineas estan dentro la zona sombreada en azul
# Esto sugiere que los coeficientes no son significativos
# por lo que es un indicador de que no existe una dependencia del tiempo en los datos
# no existe autocorrelacion
sgt.plot_pacf(dfTrain.wn, lags=40, zero=False, method="ols")
plt.title("PACF WN", size=24)
# plt.show()

# Se observa que todas las lineas son mas altas que la zona sombreada en azul
# Esto sugiere que los coeficientes son significativos
# por lo que es un indicador de la dependencia del tiempo en los datos
# existe autocorrelacion
sgt.plot_pacf(dfTrain.rw, lags=40, zero=False, method="ols")
plt.title("PACF RW", size=24)
plt.show()
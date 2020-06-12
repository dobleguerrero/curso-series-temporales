import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filePath = "../../3. Características fundamentales/Python/airline_passengers.csv"
airline = pd.read_csv(filePath, index_col="Month", parse_dates=True)

df = airline.copy()
df.columns = df.columns.str.replace(' ', '')

df.dropna(inplace=True)

df.index.freq = "MS"

# SMA: SIMPLE MOVING AVERAGE
# (x1 + x1 + ... + xn) / n

df["6SMA"] = df.ThousandsofPassengers.rolling(window=6).mean()
df["12SMA"] = df.ThousandsofPassengers.rolling(window=12).mean()

# En escenarios reales es comun cambiar los pesos de los elementos de la ventana
# es decir, par el periodo actual los meses anteriores mas cercanos deben tener
# mayor peso que los mas lejanos. Para estos casos se implementa el metodo

# EWMA: Exponentially Weighted Moving Average
# SUM(wi * xt-i) / SUM(wi)
# adjust = true: wi = (1 - a)^i
# adjust = false: si i < t: wi = a(1 - a)^i, si i = t: wi = (1 - a)^i
# el parametro de suavizado a (alpha) es un valor entre 0 y 1
# Como calcular el parametro de suavizado
# span: Promedio movil de N periodos. Para una duracion span >= 1: 2 / s + 1
# center of mass: Interpretacion fisica en terminos de duracio: c = (s - 1) / 2. 
# Para un centro de masa c >= 0: 1 / 1 + c
# half-life: Periodo de tiempo para que el peso exponencial se reduzca a la mitad.
# Para un parametro half-life h > 0: 1 - exp^(log 0.5 / h)
# alpha: Parametro como numero directo.

span = 12
alpha = 2 / (span + 1)

df["12EWMA"] = df.ThousandsofPassengers.ewm(alpha=alpha, adjust=False).mean()

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

df["12SES"] = SimpleExpSmoothing(df.ThousandsofPassengers).fit(smoothing_level=alpha, optimized=False).fittedvalues.shift(-1)

# Este es un metodo de suavizado exponencial simple que no considera los factores
# de tendencia y estacionalidad del conjunto de datos.
# Por esto se propone el uso de de suavizado exponencial doble y triple con el metodo Holt-Winters

# Hol-Winters

# Suavizado doble, Metodo de Holt: Se presenta un nuevo factor de suabizado b (beta) aborda la tendencia

# Suavizado triple, Metodo de Holt-Winters: Se presenta un nuevo factor de suavizado g (gamma) aborda la estacionalidad,
# además L representa las divisiones por ciclo, por ejemplo, en este caso los datos mensuales muestran
# un patron repetitivo cada año, entonces L = 12.

# En general los valores mas altos para alpha, beta, gamma (valores mas cercanos a 1) ponen mas enfasis en datos recientes.

# Suavizado Exponencial Doble
# Se agrega factor para la tendencia
# Para tendencias lineales se usa el metodo aditivo, para tendencias exponenciales el metodo multiplicativo
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df["12DESA"] = ExponentialSmoothing(df.ThousandsofPassengers, trend="add").fit().fittedvalues.shift(-1)

df["12DESM"] = ExponentialSmoothing(df.ThousandsofPassengers, trend="mul").fit().fittedvalues.shift(-1)

# Suavizado Exponencial Triple
# Se agrega factor para la tendencia
# Se agrega factor para la estacionalidad
# Para comportamientos lineales se usa el metodo aditivo, para comportamientos exponenciales el metodo multiplicativo
# Ademas se agrega las divisiones por ciclo L
from statsmodels.tsa.holtwinters import ExponentialSmoothing

L = 12

df["12TESA"] = ExponentialSmoothing(df.ThousandsofPassengers, trend="add", seasonal="add", seasonal_periods=L).fit().fittedvalues

df["12TESM"] = ExponentialSmoothing(df.ThousandsofPassengers, trend="mul", seasonal="mul", seasonal_periods=L).fit().fittedvalues

print(df.head(15))

modelAdd = ExponentialSmoothing(df.ThousandsofPassengers, trend="add", seasonal="add", seasonal_periods=L).fit()

modelMul = ExponentialSmoothing(df.ThousandsofPassengers, trend="mul", seasonal="mul", seasonal_periods=L).fit()

addPredict = modelAdd.forecast(36)

mulPredict = modelMul.forecast(36)

#df.plot()
df.ThousandsofPassengers.plot()
addPredict.plot()
mulPredict.plot()
plt.show()

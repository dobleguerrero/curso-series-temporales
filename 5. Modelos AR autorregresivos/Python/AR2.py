import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARMA
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts

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
df["marketValue"] = df.ftse
# eliminar columnas
del df["spx"]
del df["dax"]
del df["ftse"]
del df["nikkei"]

size = int(len(df) * 0.8)
dfTrain, dfTest = df.iloc[:size], df.iloc[size:]

# Los modelos autorregresivos funcionan mejor para series estacionarias que no estacionarias
# prueba Dickey-Fuller para saber si los datos son significativamente no estacionarios
# como el valor del estadistico de prueba (-1.904155141883686) es mayor que el valor critico
# en los distintos niveles de significacion, entonces se acepta la hipotesis nula
# por lo tanto los datos no son estacionarios
print(sts.adfuller(dfTrain.marketValue))

# En finanzas cuando los datos no son estacionarios, entonces se trabaja con datos de retorno
# ya que es el porcentaje de cambio de precios entre periodos consecutivos
dfTrain["returns"] = df.marketValue.pct_change(1).mul(100)
dfTrain = dfTrain.iloc[1:]
# print(dfTrain)

# Ahora la prueba Dickey-Fuller para saber si los datos son significativamente no estacionarios
# como el valor del estadistico de prueba (-12.77) es menor que los valores criticos
# en los distintos niveles de significacion, entonces se rechaza la hipotesis nula
# por lo tanto los datos son estacionarios
print(sts.adfuller(dfTrain.returns))

# Grafica de autocorrelacion
# la practica comun es analizar los primeros 40 retrasos
# zero=False omite la autocorrelacion consigo mismo

sgt.plot_acf(dfTrain.returns, lags=40, zero=False)
plt.title("ACF for Returns", size = 20)
# plt.show()

# AUTOCORRELACION PARCIAL
# PACF: Parcial AutoCorrelation Function
# Ignora la autocorrelacion indirecta

sgt.plot_pacf(dfTrain.returns, lags=40, alpha=0.05, zero=False, method="ols")
plt.title("PACF for Returns", size = 20)
plt.show()

# Criterios de evaluacion
# Logaritmo de la verosimilitud - Log Likelihood
# Mientras mas complejidad, este logaritmo aumenta
# AIC - BIC
# Se buscan los valores mas bajos de estos criterios

# Contraste de razon de Log verosimilitudes - Likelihood Ratio Test
# Compara dos modelos y regresa un p valor
# para saber si los modelos son significativamente similares
# si p valor < 0.05, entonces los modelos son significativamente diferentes
# por lo que al aumentar la complejidad contribuye al rendimiento del modelo
# de lo contrario los modelos son significativamente similares
# entonces se selecciona el de menor complejidad
# ya que al aumentar la complejidad no contribuye al rendimiento del modelo

# una regla ademas del p valor es que el ultimo coeficiente del modelo seleccionado
# debe ser significativamente distinto de cero, de lo contrario ese coeficiente se descarta
# y seria un modelo de menor orden

# Degree of Freedom DF = parametrosmodel1 - parametrosmodel2
def LLR_test(model1, model2, DF=1):
    L1 = model1.llf
    L2 = model2.llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p

# Modelo AR de orden 1 y sin valores residuales
# la constante y el coeficiente de retraso son significativos
modelAR1 = ARMA(dfTrain.returns, order=(1,0))
resultsAR1 = modelAR1.fit()
print(resultsAR1.summary())

# Modelo AR de orden 2 y sin valores residuales
# los coeficientes de retraso son significativos, pero la constante no
# el p valor de LLR es significativo, entonces se selecciona el modelo de orden 2
modelAR2 = ARMA(dfTrain.returns, order=(2,0))
resultsAR2 = modelAR2.fit()
print(resultsAR2.summary())
print ("LLR test: " + str(LLR_test(resultsAR1, resultsAR2)))

# Modelo AR de orden 3 y sin valores residuales
# los coeficientes de retraso son significativos, pero la constante no
# el p valor de LLR es significativo, entonces se selecciona el modelo de orden 3
modelAR3 = ARMA(dfTrain.returns, order=(3,0))
resultsAR3 = modelAR3.fit()
print(resultsAR3.summary())
print ("LLR test: " + str(LLR_test(resultsAR2, resultsAR3)))

# Modelo AR de orden 4 y sin valores residuales
# los coeficientes de retraso son significativos, pero la constante no
# el p valor de LLR es significativo, entonces se selecciona el modelo de orden 4
modelAR4 = ARMA(dfTrain.returns, order=[4,0])
resultsAR4 = modelAR4.fit()
print(resultsAR4.summary()) 
print ("LLR test: " + str(LLR_test(resultsAR3, resultsAR4)))

# Modelo AR de orden 5 y sin valores residuales
# solo el coeficiente del primer retraso y la constante no son significativos
# el p valor de LLR es significativo, entonces se selecciona el modelo de orden 5
modelAR5 = ARMA(dfTrain.returns, order=(5,0))
resultsAR5 = modelAR5.fit()
print(resultsAR5.summary())
print("LLR test p-value = " + str(LLR_test(resultsAR4, resultsAR5)))

# Modelo AR de orden 6 y sin valores residuales
# los coeficientes de retraso son significativos, pero la constante no
# el p valor de LLR es significativo, entonces se selecciona el modelo de orden 6
modelAR6 = ARMA(dfTrain.returns, order=(6,0))
resultsAR6 = modelAR6.fit()
print(resultsAR6.summary())
print("LLR test p-value = " + str(LLR_test(resultsAR5, resultsAR6)))

# Modelo AR de orden 7 y sin valores residuales
# El ultimo coeficiente no es significativo
# ademas el p valor de LLR no es significativo, entonces se selecciona el modelo anterior
modelAR7 = ARMA(dfTrain.returns, order=(7,0))
resultsAR7 = modelAR7.fit()
print(resultsAR7.summary())
print("LLR test p-value = " + str(LLR_test(resultsAR6, resultsAR7)))

# El ultimo coeficiente y el p valor de LLR son significativos, entonces se selecciona el modelo 6
# print("LLR test: " + str(LLR_test(resultsAR1, resultsAR6, DF = 5)))

# Analisis de residuos
# deben tener comportamiento semejante a ruido blanco (no deben presentar autocorrelacion)
# el test de Dickey-Fuller se debe obtener que los residuos son significativamente estacionarios
dfTrain['resPrice'] = resultsAR6.resid
print("Dickey-Fuller", sts.adfuller(dfTrain.resPrice))
sgt.plot_acf(dfTrain.resPrice, zero = False, lags = 40)
plt.title("ACF Of Residuals for Prices",size=24)
plt.show()

dfTrain.resPrice[1:].plot()
plt.title("Residuals of Prices",size=24)
plt.show()
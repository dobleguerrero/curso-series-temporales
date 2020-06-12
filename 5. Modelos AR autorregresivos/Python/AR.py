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

# Grafica de autocorrelacion
# la practica comun es analizar los primeros 40 retrasos
# zero=False omite la autocorrelacion consigo mismo

# Se observa que todas las lineas son mas altas que la zona sombreada en azul
# Esto sugiere que los coeficientes son significativos
# por lo que es un indicador de la dependencia del tiempo en los datos
# existe autocorrelacion
sgt.plot_acf(dfTrain.marketValue, lags=40, zero=False)
plt.title("ACF for Prices", size = 20)
plt.show()

# AUTOCORRELACION PARCIAL
# PACF: Parcial AutoCorrelation Function
# Ignora la autocorrelacion indirecta

# A partir del retraso 25, los coeficientes son practicamente cero por lo que no son significativos
# Despues del retraso 22 son negativos, de acuerdo a la frecuencia diaria de dias habiles
# se sabe que aproximadamente se tienen 22 dias habiles por mes entonces puede ser 
# que los valores del mes anterior afecten negativamente a los valores de hoy
# El primer retraso es bastante significativo
sgt.plot_pacf(dfTrain.marketValue, lags=40, alpha=0.05, zero=False, method="ols")
plt.title("PACF for Prices", size = 20)
plt.show()

# SUMMARY - PARTE INFERIOR
# coef - Valor de los coeficientes
# std err - Errores standar
# z - valor del estadistico de prueba
# P>|z| - P valor, Si es pequeño, entonces el coeficiente es significativo. valor comun 0.05
# [0.025 - 0.975] - Intervalo de confianza, si contiene el cero no es significativo

# Modelo AR de orden 1 y sin valores residuales
# la constante y el coeficiente de retraso son significativos
modelAR = ARMA(dfTrain.marketValue, order=(1,0))
resultsAR = modelAR.fit()
resultsAR.summary()

# Modelo AR de orden 2 y sin valores residuales
# solo la constante y el coeficiente del primer retraso son significativos
# entonces el precio de hace 2 dias no afectan en gran medida el precio de hoy
modelAR2 = ARMA(dfTrain.marketValue, order=(2,0))
resultsAR2 = modelAR2.fit()
resultsAR2.summary()

# Modelo AR de orden 3 y sin valores residuales
# solo el coeficiente del segundo retraso no es significativo
modelAR3 = ARMA(dfTrain.marketValue, order=(3,0))
resultsAR3 = modelAR3.fit()
resultsAR3.summary()

# Modelo AR de orden 4 y sin valores residuales
# los coeficientes del segundo y tercer retraso no son significativos
modelAR4 = ARMA(dfTrain.marketValue, order=[4,0])
resultsAR4 = modelAR4.fit()
resultsAR4.summary()

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

# p = 0.001, entonces se selecciona el modelo 3
LLR_test(resultsAR2, resultsAR3)
# p = 0.0, entonces se selecciona el modelo 4
LLR_test(resultsAR3, resultsAR4)

modelAR4 = ARMA(dfTrain.marketValue, order=[4,0])
resultsAR4 = modelAR4.fit()
print(resultsAR4.summary()) 
print ("LLR test: " + str(LLR_test(resultsAR3, resultsAR4)))

# El ultimo coeficiente del modelo es significativamente diferente de cero
# el p valor de LLR es 0.04, entonces se selecciona este modelo
modelAR5 = ARMA(dfTrain.marketValue, order=(5,0))
resultsAR5 = modelAR5.fit()
print(resultsAR5.summary())
print("\nLLR test p-value = " + str(LLR_test(resultsAR4, resultsAR5)))

# El ultimo coeficiente y el p valor de LLR son significativos, entonces se selecciona este modelo
modelAR6 = ARMA(dfTrain.marketValue, order=(6,0))
resultsAR6 = modelAR6.fit()
print(resultsAR6.summary())
print("\nLLR test p-value = " + str(LLR_test(resultsAR5, resultsAR6)))

# El ultimo coeficiente y el p valor de LLR son significativos, entonces se selecciona este modelo
modelAR7 = ARMA(dfTrain.marketValue, order=(7,0))
resultsAR7 = modelAR7.fit()
print(resultsAR7.summary())
print("\nLLR test p-value = " + str(LLR_test(resultsAR6, resultsAR7)))

# El ultimo coeficiente no es significativo
# ademas el p valor de LLR no es significativo, entonces se selecciona el modelo anterior
modelAR8 = ARMA(dfTrain.marketValue, order=(8,0))
resultsAR8 = modelAR8.fit()
print(resultsAR8.summary())
print("\nLLR test p-value = " + str(LLR_test(resultsAR7, resultsAR8)))

# El ultimo coeficiente y el p valor de LLR son significativos, entonces se selecciona el modelo 7
print("LLR test: " + str(LLR_test(resultsAR, resultsAR7, DF = 6)))

# Los modelos autorregresivos funcionan mejor para series estacionarias que no estacionarias

# Analisis de residuos
# deben tener comportamiento semejante a ruido blanco (no deben presentar autocorrelacion)
# el test de Dickey-Fuller se debe obtener que los residuos son significativamente estacionarios
dfTrain['resPrice'] = resultsAR7.resid
sts.adfuller(dfTrain.resPrice)
sgt.plot_acf(dfTrain.resPrice, zero = False, lags = 40)
plt.title("ACF Of Residuals for Prices",size=24)
plt.show()

dfTrain.resPrice[1:].plot()
plt.title("Residuals of Prices",size=24)
plt.show()
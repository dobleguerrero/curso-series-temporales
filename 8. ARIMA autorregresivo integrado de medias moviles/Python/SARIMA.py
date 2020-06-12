import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Herramientas
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose      # ETS Plots
                            
# Warnings
import warnings
warnings.filterwarnings("ignore")
# Dataset
df = pd.read_csv('./co2_mm_mlo.csv')
df.head()

# Añadir una variable "date"
df['date']=pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))
# Hacer que la variable "date" sea el indice
df.set_index('date',inplace=True)
df.index.freq = 'MS'
df.head()

title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
ylabel='parts per million'
xlabel='' 

ax = df['interpolated'].plot(figsize=(12,6),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)

# Descomposición en 3 factores
result = seasonal_decompose(df['interpolated'], model='add')
result.plot()

# Aunque sea pequeña en comparación con la escala de los datos, hay una estacionalidad anual.
# Dividir en datos de entrenamiento y prueba
len(df)
train = df.iloc[:717]
test = df.iloc[717:]

# Modelo SARIMA(0,1,1)(1,0,1,12) 
model = SARIMAX(train['interpolated'],order=(0,1,1),seasonal_order=(1,0,1,12))
results = model.fit()
results.summary()

# Obtain predicted values
start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end).rename('SARIMA(0,1,1)(1,0,1,12) Predictions')

# Compare predictions to expected values
for i in range(len(predictions)):
    print(f"predicted={predictions[i]:<11.10}, expected={test['interpolated'][i]}")

# Plot predictions against known values
title ='Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
ylabel='parts per million'
xlabel=''
ax = test['interpolated'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)

# Reentrenar el modelo con todos los datos y predecir el futuro
model = SARIMAX(df['interpolated'],order=(0,1,1),seasonal_order=(1,0,1,12))
results = model.fit()
fcast = results.predict(len(df),len(df)+36).rename('SARIMA(0,1,1)(1,0,1,12) Forecast')

# Plot predictions against known values
title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
ylabel='parts per million'
xlabel=''
ax = df['interpolated'].plot(legend=True,figsize=(12,6),title=title)
fcast.plot(legend=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
plt.show()
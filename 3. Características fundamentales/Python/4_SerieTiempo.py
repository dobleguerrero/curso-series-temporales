import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## Componentes de una serie de tiempo Hodrick-Prescot
# Tendencia: Comportamiento de los datos
# Estacionalidad: Presencia de tendencias repetitivas
# Ciclico: Presencia de tendencias sin repeticion establecida

# Este filtro depende de un parametro lambda
# valores comunes de acuerdo a la frecuencia
# Trimestrales: 1600
# Anuales: 6.25
# Mensuales: 129600

archivo = "macrodata.csv"
df = pd.read_csv(archivo, index_col=0, parse_dates=True)
df = df.copy()
# ax = df.realgdp.plot()
# ax.autoscale(axis="x", tight=True)
# ax.set(ylabel="REAL GDP")
# plt.show()

# DESCOMPOSICION ADITIVA
# from statsmodels.tsa.seasonal import seasonal_decompose
# additive = seasonal_decompose(df.realgdp, model="additive")
# additive.plot()
# plt.show()

from statsmodels.tsa.filters.hp_filter import hpfilter
gdpCycle, gdpTrend = hpfilter(df.realgdp, lamb=1600)

df["Trend"] = gdpTrend

# df[["Trend", "realgdp"]].plot(figsize=(10,5))
# plt.show()

df[["Trend", "realgdp"]]["2005-01-01":].plot(figsize=(10,5))
plt.show()
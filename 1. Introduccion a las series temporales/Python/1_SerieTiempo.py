import pandas as pd
import numpy as np

archivo = "Index2018.csv"
raw_csv_data = pd.read_csv(archivo)

df = raw_csv_data.copy()

print(df.head())

print(df.describe())

print(df.isna().sum())

print(df.spx.isna().sum())

# TIMESERIES PLOT
import matplotlib.pyplot as plt

df.spx.plot(figsize=(10, 5), title="SP500 Prices")
plt.show()

df.ftse.plot(figsize=(10, 5), title="FTSE100 Prices")
plt.show()

df.spx.plot(figsize=(10, 5))
df.ftse.plot(figsize=(10, 5))
plt.title("SPX vs FTSE")
plt.show()

# Quantile-Quantile (QQ) PLOT
import scipy.stats

# QQ plot explica la distribucion de los datos
# para saber si se ajustan a una distribucion normal, por ejemplo
# miden a cuantas desviaciones estandar de la media estan los datos
# la linea diagonal, representa la distribucion normal
# En este caso los datos no se distribuyen de una forma normal
scipy.stats.probplot(df.spx, plot=plt)
plt.title("QQ plot", size=24)
plt.show()

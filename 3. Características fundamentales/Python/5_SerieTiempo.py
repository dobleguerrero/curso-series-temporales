import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filePath = "airline_passengers.csv"
airline = pd.read_csv(filePath, index_col="Month", parse_dates=True)

df = airline.copy()
df.columns = df.columns.str.replace(' ', '')
print(df)

df.plot()
# plt.show()

# Si la tendencia es lineal,
# la estacionalidad y la tendencia parecen ser constantes
# entonces se aplica el modelo aditivo
# si no se trata de un comportamiento lineal
# entonces se aplica el modelo multiplicativo
from statsmodels.tsa.seasonal import seasonal_decompose
decompose = seasonal_decompose(df.ThousandsofPassengers, model="Aditive")
decompose.plot()
# plt.show()

decompose = seasonal_decompose(df.ThousandsofPassengers, model="Multiplicative")
decompose.plot()
plt.show()

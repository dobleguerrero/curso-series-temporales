import pandas as pd
import numpy as np

archivo = "Index2018.csv"
raw_csv_data = pd.read_csv(archivo)

df = raw_csv_data.copy()

print(df.date.describe())

df.date = pd.to_datetime(df.date, dayfirst=True)

print(df.date.describe())

df.set_index("date", inplace=True)

print(df.head())

# h: hourly, w: weekly, d: daily, m: monthly, a: anual, b: business days
df = df.asfreq("b")

print(df.head())

print(df.isna().sum())

# front filling, asigna el valor del periodo posterior
df.spx = df.spx.fillna(method="ffill")
# back filling, asigna el valor del periodo anterior
df.ftse = df.ftse.fillna(method="bfill")
df.nikkei = df.nikkei.fillna(method="bfill")
# asignar valor constante
# para series de tiempo no aplica en la mayoria de los casos
df.dax = df.dax.fillna(value=df.dax.mean())

print(df.isna().sum())

df["market_value"] = df.spx

print(df.describe())
# eliminar columnas
del df["spx"]
del df["dax"]
del df["ftse"]
del df["nikkei"]

print(df.describe())

size = int(len(df) * 0.8)

dfTrain = df.iloc[:size]
dfTest = df.iloc[size:]

print(dfTrain.tail())
print(dfTest.head())
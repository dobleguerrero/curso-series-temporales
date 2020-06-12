import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARMA
from scipy.stats.distributions import chi2 
from math import sqrt
# import seaborn as sns
# sns.set()

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

# Degree of Freedom DF = parametrosmodel1 - parametrosmodel2
def LLR_test(model1, model2, DF=1):
    L1 = model1.llf
    L2 = model2.llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p

import warnings
warnings.filterwarnings("ignore")

dfTrain["returns"] = dfTrain.marketValue.pct_change(1).mul(100)
dfTrain = dfTrain.iloc[1:]

sgt.plot_acf(dfTrain.returns[1:], zero = False, lags = 40)
plt.title("ACF for Returns", size=24)
plt.show()

sgt.plot_pacf(dfTrain.returns[1:], lags = 40, zero = False, method = ('ols'))
plt.title("PACF FTSE Returns", size=24)
plt.show()

model_ret_ma_1 = ARMA(dfTrain.returns[1:], order=(0,1))
results_ret_ma_1 = model_ret_ma_1.fit()
results_ret_ma_1.summary()

model_ret_ma_2 = ARMA(dfTrain.returns[1:], order=(0,2))
results_ret_ma_2 = model_ret_ma_2.fit()
print(results_ret_ma_2.summary())
print("\nLLR test p-value = " + str(LLR_test(results_ret_ma_1, results_ret_ma_2)))

model_ret_ma_3 = ARMA(dfTrain.returns[1:], order=(0,3))
results_ret_ma_3 = model_ret_ma_3.fit()
print(results_ret_ma_3.summary())
print("\nLLR test p-value = " + str(LLR_test(results_ret_ma_2, results_ret_ma_3)))

model_ret_ma_4 = ARMA(dfTrain.returns[1:], order=[0,4])
results_ret_ma_4 = model_ret_ma_4.fit()
print(results_ret_ma_4.summary())
print("\nLLR test p-value = " + str(LLR_test(results_ret_ma_3, results_ret_ma_4)))

model_ret_ma_5 = ARMA(dfTrain.returns[1:], order=[0,5])
results_ret_ma_5 = model_ret_ma_5.fit()
print(results_ret_ma_5.summary())
print("\nLLR test p-value = " + str(LLR_test(results_ret_ma_4, results_ret_ma_5)))

model_ret_ma_6 = ARMA(dfTrain.returns[1:], order=[0,6])
results_ret_ma_6 = model_ret_ma_6.fit()
print(results_ret_ma_6.summary())
print("\nLLR test p-value = " + str(LLR_test(results_ret_ma_5, results_ret_ma_6)))

model_ret_ma_7 = ARMA(dfTrain.returns[1:], order=[0,7])
results_ret_ma_7 = model_ret_ma_7.fit()
print(results_ret_ma_7.summary())
print("\nLLR test p-value = " + str(LLR_test(results_ret_ma_6, results_ret_ma_7)))

model_ret_ma_8 = ARMA(dfTrain.returns[1:], order=[0,8])
results_ret_ma_8 = model_ret_ma_8.fit()
print(results_ret_ma_8.summary())
print("\nLLR test p-value = " + str(LLR_test(results_ret_ma_7, results_ret_ma_8)))

LLR_test(results_ret_ma_6, results_ret_ma_8, DF = 2)

dfTrain['res_ret_ma_8'] = results_ret_ma_8.resid[1:]

print("The mean of the residuals is " + str(round(dfTrain.res_ret_ma_8.mean(),3)) + "\nThe variance of the residuals is " + str(round(dfTrain.res_ret_ma_8.var(),3)))

round(sqrt(dfTrain.res_ret_ma_8.var()),3)

dfTrain.res_ret_ma_8[1:].plot(figsize = (20,5))
plt.title("Residuals of Returns", size = 24)
plt.show()

sts.adfuller(dfTrain.res_ret_ma_8[2:])

sgt.plot_acf(dfTrain.res_ret_ma_8[2:], zero = False, lags = 40)
plt.title("ACF Of Residuals for Returns",size=24)
plt.show()

benchmark = dfTrain.marketValue.iloc[0]
dfTrain['norm'] = dfTrain.marketValue.div(benchmark).mul(100)

sts.adfuller(dfTrain['norm'])

bench_ret = dfTrain.returns.iloc[1]
dfTrain['norm_ret'] = dfTrain.returns.div(bench_ret).mul(100)

sts.adfuller(dfTrain.norm_ret[1:])

sgt.plot_acf(dfTrain.norm_ret[1:], zero = False, lags = 40)
plt.title("ACF of Normalized Returns",size=24)
plt.show()

model_norm_ret_ma_8 = ARMA(dfTrain.norm_ret[1:], order=(0,8))
results_norm_ret_ma_8 = model_norm_ret_ma_8.fit()
results_norm_ret_ma_8.summary()

dfTrain['res_norm_ret_ma_8'] = results_norm_ret_ma_8.resid[1:]

sts.adfuller(dfTrain.res_norm_ret_ma_8[2:])

dfTrain.res_norm_ret_ma_8[1:].plot(figsize=(20,5))
plt.title("Residuals of Normalized Returns",size=24)
plt.show()

sgt.plot_acf(dfTrain.res_norm_ret_ma_8[2:], zero = False, lags = 40)
plt.title("ACF Of Residuals for Normalized Returns",size=24)
plt.show()

sgt.plot_acf(dfTrain.marketValue, zero = False, lags = 40)
plt.title("ACF for Prices", size=20)
plt.show()

model_ma_1 = ARMA(dfTrain.marketValue, order=(0,1))
results_ma_1 = model_ma_1.fit()
results_ma_1.summary()
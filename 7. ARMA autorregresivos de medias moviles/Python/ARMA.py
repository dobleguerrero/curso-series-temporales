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

model_ret_ar_1_ma_1 = ARMA(dfTrain.returns[1:], order=(1,1))
results_ret_ar_1_ma_1 = model_ret_ar_1_ma_1.fit()
results_ret_ar_1_ma_1.summary()

model_ret_ar_1 = ARMA(dfTrain.returns[1:], order = (1,0))
model_ret_ma_1 = ARMA(dfTrain.returns[1:], order = (0,1))
results_ret_ar_1 = model_ret_ar_1.fit()
results_ret_ma_1 = model_ret_ma_1.fit()
print("\nARMA vs AR ", LLR_test(results_ret_ar_1, results_ret_ar_1_ma_1))
print("\nARMA vs MA ", LLR_test(results_ret_ma_1, results_ret_ar_1_ma_1))

sgt.plot_acf(dfTrain.returns[1:], zero = False, lags = 40)
plt.title("ACF for Returns",size=24)
plt.show()

sgt.plot_pacf(dfTrain.returns[1:], zero = False, lags = 40, method='ols')
plt.title("PACF for Returns",size=24)
plt.show()

model_ret_ar_3_ma_3 = ARMA(dfTrain.returns[1:], order=(3,3))
results_ret_ar_3_ma_3 = model_ret_ar_3_ma_3.fit()
results_ret_ar_3_ma_3.summary()

LLR_test(results_ret_ar_1_ma_1, results_ret_ar_3_ma_3, DF = 4)

model_ret_ar_3_ma_2 = ARMA(dfTrain.returns[1:], order=(3,2))
results_ret_ar_3_ma_2 = model_ret_ar_3_ma_2.fit()
results_ret_ar_3_ma_2.summary()

model_ret_ar_2_ma_3 = ARMA(dfTrain.returns[1:], order=(2,3))
results_ret_ar_2_ma_3 = model_ret_ar_2_ma_3.fit()
results_ret_ar_2_ma_3.summary()

model_ret_ar_3_ma_1 = ARMA(dfTrain.returns[1:], order=(3,1))
results_ret_ar_3_ma_1 = model_ret_ar_3_ma_1.fit()
results_ret_ar_3_ma_1.summary()

LLR_test(results_ret_ar_3_ma_1, results_ret_ar_3_ma_2)

model_ret_ar_2_ma_2 = ARMA(dfTrain.returns[1:], order=(2,2))
results_ret_ar_2_ma_2 = model_ret_ar_2_ma_2.fit()
results_ret_ar_2_ma_2.summary()

model_ret_ar_1_ma_3 = ARMA(dfTrain.returns[1:], order=(1,3))
results_ret_ar_1_ma_3 = model_ret_ar_1_ma_3.fit()
results_ret_ar_1_ma_3.summary()

print("\n ARMA(3,2): \tLL = ", results_ret_ar_3_ma_2.llf, "\tAIC = ", results_ret_ar_3_ma_2.aic)
print("\n ARMA(1,3): \tLL = ", results_ret_ar_1_ma_3.llf, "\tAIC = ", results_ret_ar_1_ma_3.aic)

dfTrain['res_ret_ar_3_ma_2'] = results_ret_ar_3_ma_2.resid[1:]

dfTrain.res_ret_ar_3_ma_2.plot(figsize = (20,5))
plt.title("Residuals of Returns", size=24)
plt.show()

sgt.plot_acf(dfTrain.res_ret_ar_3_ma_2[2:], zero = False, lags = 40)
plt.title("ACF Of Residuals for Returns",size=24)
plt.show()

model_ret_ar_5_ma_5 = ARMA(dfTrain.returns[1:], order=(5,5))
results_ret_ar_5_ma_5 = model_ret_ar_5_ma_5.fit()
results_ret_ar_5_ma_5.summary()

model_ret_ar_5_ma_1 = ARMA(dfTrain.returns[1:], order=(5,1))
results_ret_ar_5_ma_1 = model_ret_ar_5_ma_1.fit()
results_ret_ar_5_ma_1.summary()

model_ret_ar_1_ma_5 = ARMA(dfTrain.returns[1:], order=(1,5))
results_ret_ar_1_ma_5 = model_ret_ar_1_ma_5.fit()
results_ret_ar_1_ma_5.summary()

print("ARMA(5,1):  \t LL = ",results_ret_ar_5_ma_1.llf,"\t AIC = ",results_ret_ar_5_ma_1.aic)
print("ARMA(1,5):  \t LL = ",results_ret_ar_1_ma_5.llf,"\t AIC = ",results_ret_ar_1_ma_5.aic)

print("ARMA(3,2):  \t LL = ",results_ret_ar_3_ma_2.llf,"\t AIC = ",results_ret_ar_3_ma_2.aic)

dfTrain['res_ret_ar_5_ma_1'] = results_ret_ar_5_ma_1.resid

sgt.plot_acf(dfTrain.res_ret_ar_5_ma_1[1:], zero = False, lags = 40)
plt.title("ACF of Residuals for Returns",size=24)
plt.show()

sgt.plot_acf(dfTrain.marketValue, unbiased=True, zero = False, lags = 40)
plt.title("Autocorrelation Function for Prices",size=20)
plt.show()

sgt.plot_pacf(dfTrain.marketValue, lags = 40, alpha = 0.05, zero = False , method = ('ols'))
plt.title("Partial Autocorrelation Function for Prices",size=20)
plt.show()

model_ar_1_ma_1 = ARMA(dfTrain.marketValue, order=(1,1))
results_ar_1_ma_1 = model_ar_1_ma_1.fit()
results_ar_1_ma_1.summary()

dfTrain['res_ar_1_ma_1'] = results_ar_1_ma_1.resid

sgt.plot_acf(dfTrain.res_ar_1_ma_1, zero = False, lags = 40)
plt.title("ACF Of Residuals of Prices",size=20)
plt.show()

model_ar_6_ma_6 = ARMA(dfTrain.marketValue, order=(6,6))
#results_ar_6_ma_6 = model_ar_6_ma_6.fit()
results_ar_6_ma_6 = model_ar_6_ma_6.fit(start_ar_lags = 11)
results_ar_6_ma_6.summary()

model_ar_6_ma_2 = ARMA(dfTrain.marketValue, order=(6,2))
results_ar_6_ma_2 = model_ar_6_ma_2.fit()
results_ar_6_ma_2.summary()

dfTrain['res_ar_6_ma_2'] = results_ar_6_ma_2.resid
sgt.plot_acf(dfTrain.res_ar_6_ma_2, zero = False, lags = 40)
plt.title("ACF Of Residuals of Prices",size=20)
plt.show()

print("ARMA(6,2):  \t LL = ", results_ar_6_ma_2.llf, "\t AIC = ", results_ar_6_ma_2.aic)
print("ARMA(5,1):  \t LL = ", results_ret_ar_5_ma_1.llf, "\t AIC = ", results_ret_ar_5_ma_1.aic)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2 
from math import sqrt
# import seaborn as sns
# sns.set()

archivo = "Index2018.csv"
rawData = pd.read_csv(archivo)

dfTrain = rawData.copy()
dfTrain.date = pd.to_datetime(dfTrain.date, dayfirst=True)
dfTrain.set_index("date", inplace=True)
# h: hourly, w: weekly, d: daily, m: monthly, a: anual, b: business days
df = dfTrain.asfreq("b")
# ffill: front filling, asigna el valor del periodo posterior
# bfill: back filling, asigna el valor del periodo anterior
# asignar valor constante
# para series de tiempo no aplica en la mayoria de los casos
df = dfTrain.fillna(method="ffill")
dfTrain["marketValue"] = dfTrain.ftse
# eliminar columnas
# del dfTrain["spx"]
# del dfTrain["dax"]
# del dfTrain["ftse"]
# del dfTrain["nikkei"]

size = int(len(df) * 0.8)
dfTrain, dfTest = dfTrain.iloc[:size], dfTrain.iloc[size:]

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
# dfTrain = dfTrain.iloc[1:]

model_ar_1_i_1_ma_1 = ARIMA(dfTrain.marketValue, order=(1,1,1))
results_ar_1_i_1_ma_1 = model_ar_1_i_1_ma_1.fit()
results_ar_1_i_1_ma_1.summary()

dfTrain['res_ar_1_i_1_ma_1'] = results_ar_1_i_1_ma_1.resid
sgt.plot_acf(dfTrain.res_ar_1_i_1_ma_1, zero = False, lags = 40)
plt.title("ACF Of Residuals for ARIMA(1,1,1)",size=20)
plt.show()

dfTrain['res_ar_1_i_1_ma_1'] = results_ar_1_i_1_ma_1.resid.iloc[:]
sgt.plot_acf(dfTrain.res_ar_1_i_1_ma_1[1:], zero = False, lags = 40)
plt.title("ACF Of Residuals for ARIMA(1,1,1)",size=20)
plt.show()

model_ar_1_i_1_ma_2 = ARIMA(dfTrain.marketValue, order=(1,1,2))
results_ar_1_i_1_ma_2 = model_ar_1_i_1_ma_2.fit()
model_ar_1_i_1_ma_3 = ARIMA(dfTrain.marketValue, order=(1,1,3))
results_ar_1_i_1_ma_3 = model_ar_1_i_1_ma_3.fit()
model_ar_2_i_1_ma_1 = ARIMA(dfTrain.marketValue, order=(2,1,1))
results_ar_2_i_1_ma_1 = model_ar_2_i_1_ma_1.fit()
model_ar_3_i_1_ma_1 = ARIMA(dfTrain.marketValue, order=(3,1,1))
results_ar_3_i_1_ma_1 = model_ar_3_i_1_ma_1.fit()
model_ar_3_i_1_ma_2 = ARIMA(dfTrain.marketValue, order=(3,1,2))
results_ar_3_i_1_ma_2 = model_ar_3_i_1_ma_2.fit(start_ar_lags=5)

print("ARIMA(1,1,1):  \t LL = ", results_ar_1_i_1_ma_1.llf, "\t AIC = ", results_ar_1_i_1_ma_1.aic)
print("ARIMA(1,1,2):  \t LL = ", results_ar_1_i_1_ma_2.llf, "\t AIC = ", results_ar_1_i_1_ma_2.aic)
print("ARIMA(1,1,3):  \t LL = ", results_ar_1_i_1_ma_3.llf, "\t AIC = ", results_ar_1_i_1_ma_3.aic)
print("ARIMA(2,1,1):  \t LL = ", results_ar_2_i_1_ma_1.llf, "\t AIC = ", results_ar_2_i_1_ma_1.aic)
print("ARIMA(3,1,1):  \t LL = ", results_ar_3_i_1_ma_1.llf, "\t AIC = ", results_ar_3_i_1_ma_1.aic)
print("ARIMA(3,1,2):  \t LL = ", results_ar_3_i_1_ma_2.llf, "\t AIC = ", results_ar_3_i_1_ma_2.aic)

print("\nLLR test p-value = " + str(LLR_test(results_ar_1_i_1_ma_2, results_ar_1_i_1_ma_3)))

print("\nLLR test p-value = " + str(LLR_test(results_ar_1_i_1_ma_1, results_ar_1_i_1_ma_3, DF = 2)))

dfTrain['res_ar_1_i_1_ma_3'] = results_ar_1_i_1_ma_3.resid
sgt.plot_acf(dfTrain.res_ar_1_i_1_ma_3[1:], zero = False, lags = 40)
plt.title("ACF Of Residuals for ARIMA(1,1,3)", size=20)
plt.show()

model_ar_5_i_1_ma_1 = ARIMA(dfTrain.marketValue, order=(5,1,1))
results_ar_5_i_1_ma_1 = model_ar_5_i_1_ma_1.fit(start_ar_lags=11)
results_ar_5_i_1_ma_1.summary()

print("ARIMA(1,1,3):  \t LL = ", results_ar_1_i_1_ma_3.llf, "\t AIC = ", results_ar_1_i_1_ma_3.aic)
print("ARIMA(5,1,1):  \t LL = ", results_ar_5_i_1_ma_1.llf, "\t AIC = ", results_ar_5_i_1_ma_1.aic)

dfTrain['res_ar_5_i_1_ma_1'] = results_ar_5_i_1_ma_1.resid
sgt.plot_acf(dfTrain.res_ar_5_i_1_ma_1[1:], zero = False, lags = 40)
plt.title("ACF Of Residuals for ARIMA(5,1,1)", size=20)
plt.show()

dfTrain['delta_prices']=dfTrain.marketValue.diff(1)

model_delta_ar_1_i_1_ma_1 = ARIMA(dfTrain.delta_prices[1:], order=(1,0,1))
results_delta_ar_1_i_1_ma_1 = model_delta_ar_1_i_1_ma_1.fit()
results_delta_ar_1_i_1_ma_1.summary()

model_ar_1_i_1_ma_1 = ARIMA(dfTrain.marketValue, order=(1,1,1))
results_ar_1_i_1_ma_1 = model_ar_1_i_1_ma_1.fit()
results_ar_1_i_1_ma_1.summary()

sts.adfuller(dfTrain.delta_prices[1:])

model_ar_1_i_1_ma_1_Xspx = ARIMA(dfTrain.marketValue, exog = dfTrain.spx, order=(1,1,1))
results_ar_1_i_1_ma_1_Xspx = model_ar_1_i_1_ma_1_Xspx.fit()
results_ar_1_i_1_ma_1_Xspx.summary()

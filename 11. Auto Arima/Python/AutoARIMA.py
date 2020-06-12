# AUTO ARIMA
# De acuerdo a los criterios de AIC y BIC
# si aumenta la bondad de ajuste del modelo (verosimilitud)
# entonces AIC y BIC disminuye

import numpy as np
import pandas as pd
# import scipy
# import statsmodels.api as sm
import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn
# from statsmodels.tsa.arima_model import ARIMA
# from arch import arch_model
import yfinance

rawData = yfinance.download(tickers="^GSPC ^FTSE ^N225 ^GDAXI", start="1994-01-07", end="2020-03-20",
                            interval="1d", group_by="ticker", auto_adjust=True, treads=True)

df = rawData.copy()

df["spx"] = df["^GSPC"].Close[:]
df["ftse"] = df["^FTSE"].Close[:]
df["nikkei"] = df["^N225"].Close[:]
df["dax"] = df["^GDAXI"].Close[:]

del df["^GSPC"]
del df["^FTSE"]
del df["^N225"]
del df["^GDAXI"]

df = df.asfreq("b")
df = df.fillna(method="ffill")

df["retSpx"] = df.spx.pct_change(1) * 100
df["retFtse"] = df.ftse.pct_change(1) * 100
df["retNikkei"] = df.nikkei.pct_change(1) * 100
df["retDax"] = df.dax.pct_change(1) * 100

# print(rawData)
# print(df)
size = int(len(df) * 0.8)

dfTrain, dfTest = df.iloc[:size], df.iloc[size:]

# print(dfTrain)
# print(dfTest)

from pmdarima import auto_arima
model = auto_arima(dfTrain.retFtse[1:])

print(model)
print(model.summary())

model2 = auto_arima(df.retFtse[1:], exogenous = df[["retSpx", "retDax", "retNikkei"]][1:],
                    m = 5, max_order = None, max_p = 7, max_q = 7, max_d = 2, max_P = 4, max_Q = 4, max_D = 2,
                    maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'oob',
                    out_of_sample_size = int(len(df)*0.2))

# exogenous -> outside factors (e.g other time series)
# m -> seasonal cycle length
# max_order -> maximum amount of variables to be used in the regression (p + q)
# max_p -> maximum AR components
# max_q -> maximum MA components
# max_d -> maximum Integrations
# maxiter -> maximum iterations we're giving the model to converge the coefficients (becomes harder as the order increases)
# alpha -> level of significance, default is 5%, which we should be using most of the time
# n_jobs -> how many models to fit at a time (-1 indicates "as many as possible")
# trend -> "ct" usually
# information_criterion -> 'aic', 'aicc', 'bic', 'hqic', 'oob' 
#        (Akaike Information Criterion, Corrected Akaike Information Criterion,
#        Bayesian Information Criterion, Hannan-Quinn Information Criterion, or
#        "out of bag"--for validation scoring--respectively)
# out_of_smaple_size -> validates the model selection (pass the entire dataset, and set 20% to be the out_of_sample_size)

print(model2)
print(model2.summary())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load specific forecasting tools
from statsmodels.tsa.ar_model import AR,ARResults

# Load the U.S. Population dataset
archivo = "uspopulation.csv"
rawData = pd.read_csv(archivo, index_col='DATE', parse_dates=True)

df = rawData.copy()
df.index.freq = 'MS'

title='U.S. Monthly Population Estimates'
ylabel='Pop. Est. (thousands)'

ax = df['PopEst'].plot(figsize=(12,5),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(ylabel=ylabel)
plt.show()

len(df)
train = df.iloc[:84]
test = df.iloc[84:]

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

model = AR(train['PopEst'])
AR1fit = model.fit(maxlag=1,method='mle')
print(f'Lag: {AR1fit.k_ar}')
print(f'Coefficients:\n{AR1fit.params}')

start=len(train)
end=len(train)+len(test)-1
predictions1 = AR1fit.predict(start=start, end=end, dynamic=False).rename('AR(1) Predictions')
print(predictions1)

# Comparando predicciones con valores reales
for i in range(len(predictions1)):
    print(f"predicted={predictions1[i]:<11.10}, expected={test['PopEst'][i]}")

test['PopEst'].plot(legend=True)
predictions1.plot(legend=True,figsize=(12,6))
plt.show()

# Recall that our model was already created above based on the training set
model = AR(train['PopEst'])
AR2fit = model.fit(maxlag=2,method='mle')
print(f'Lag: {AR2fit.k_ar}')
print(f'Coefficients:\n{AR2fit.params}')

start=len(train)
end=len(train)+len(test)-1
predictions2 = AR2fit.predict(start=start, end=end, dynamic=False).rename('AR(2) Predictions')

test['PopEst'].plot(legend=True)
predictions1.plot(legend=True)
predictions2.plot(legend=True,figsize=(12,6))
plt.show()

model = AR(train['PopEst'])
ARfit = model.fit(ic='bic')
print(f'Lag: {ARfit.k_ar}')
print(f'Coefficients:\n{ARfit.params}')

start = len(train)
end = len(train)+len(test)-1
rename = f'AR(8) Predictions'

predictions8 = ARfit.predict(start=start,end=end,dynamic=False).rename(rename)

test['PopEst'].plot(legend=True)
predictions1.plot(legend=True)
predictions2.plot(legend=True)
predictions8.plot(legend=True,figsize=(12,6))
plt.show()

from sklearn.metrics import mean_squared_error

labels = ['AR(1)','AR(2)','AR(8)']
preds = [predictions1, predictions2, predictions8]  # these are variables, not strings!
for i in range(3):
    error = mean_squared_error(test['PopEst'], preds[i])
    print(f'{labels[i]} Error: {error:11.10}')


modls = [AR1fit,AR2fit,ARfit]
for i in range(3):
    print(f'{labels[i]} AIC: {modls[i].aic:6.5}')

# First, retrain the model on the full dataset
model = AR(df['PopEst'])
# Next, fit the model
ARfit = model.fit(maxlag=8,method='mle')
# Make predictions
fcast = ARfit.predict(start=len(df), end=len(df)+12, dynamic=False).rename('Forecast')
# Plot the results
df['PopEst'].plot(legend=True)
fcast.plot(legend=True,figsize=(12,6))
plt.show()
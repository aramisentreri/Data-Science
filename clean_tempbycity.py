import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import statsmodels.api as sm

tbc_data = pd.read_csv('GlobalLandTemperaturesByCity.csv', parse_dates = True)

tempbycity = pd.DataFrame(tbc_data)
time = pd.to_datetime(tempbycity['dt'])

tempbycity['year'] = time.dt.year
tempbycity['month']= time.dt.month

clean_tempbycity = tempbycity.dropna()

santiago_data = clean_tempbycity[(clean_tempbycity.City == 'Santiago') & (clean_tempbycity.Country == 'Chile')]

grouped = santiago_data['AverageTemperature'].groupby(santiago_data['year'])
mean_temps = grouped.mean()

# Remove the first data point because it had incomplete measurments for the year, which might affect the average
# remember they are indexed now by the groupby parameter = year
mean_temps = mean_temps[mean_temps.index[1:]]

## Plotting

plt.plot(mean_temps)
plt.title('Yearly mean temperatures in Santiago, Chile')
plt.xlabel('Years')
plt.ylabel('Temperature')

## Find the hottest year and the coldest

(hot_temp,hot_year) = (mean_temps.max(),mean_temps.argmax())
(cold_temp,cold_year) = (mean_temps.min(),mean_temps.argmin())

## Eliminate local (5 years) variation

smooth_temps = mean_temps.rolling(5).mean()
plt.plot(smooth_temps)
plt.scatter(smooth_temps.index, smooth_temps)
plt.xlabel('Years')
plt.ylabel('Temperature')
plt.title('Average temp per year smoothed over 5 year period')

## Linear regression? 

smooth_temps = smooth_temps[smooth_temps.index[4:]] # removing the NaN

model = sm.OLS(smooth_temps,smooth_temps.index)
results = model.fit()

y = results.params['x1']*smooth_temps + 0.003
plt.plot(smooth_temps.index, results.fittedvalues, 'r--')

for i in range(2014,2018):
	y = y.append(pd.Series([results.params['x1']*i + 0.003], index=[i]))

plt.plot(y, 'bo',label='predicted')

## Monthly averages and predictions

aux=pd.Series()
for i in range(1,13):
	plt.plot(santiago_data['year'], santigo_data[santiago_data['month']==i]['AverageTemperature'])
	mod = sm.OLS(santiago_data[santiago_data['month']==i]['AverageTemperature'], santiago_data['year'])
	res = mod.fit()
	aux = aux.append(pd.Series([res.params['x1'],res.params['const']],index=[i]))




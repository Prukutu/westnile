import pickle

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns

def dateConverter(date, fmt='%Y-%m-%d'):
    """ Convert a string formatted date to a datetime object. """
    return datetime.strptime(date, fmt)


def fillMissing(df, col):
    # Fill missing values with the mean of the values before and after.
    for position in df[df[col] == 'M'].index.values:
        df.at[position, col] = ((float(df.loc[position-1, col]) +
                                    float(df.loc[position+1, col]) / 2))

def loadWeather(filename):

    # We only want certain features from the weather file.
    # We can always change this later.

    weather_features = ['Station', 'Date', 'Tmax', 'Tmin',
                        'Tavg', 'DewPoint', 'PrecipTotal', 'Heat',
                        'WetBulb', 'StnPressure', 'ResultSpeed',
                        'ResultDir', 'AvgSpeed']
    weather = pd.read_csv(filename)

    weather['Date'] = pd.to_datetime(weather['Date'])



    # Add the dew point difference (a measure of how moist air is)

    #
    # weather.drop(columns='Station', inplace=True)
    #
    # A value of T in precipitation indicates only trace amounts. Substitute
    # for numerical number.
    #
    weather['PrecipTotal'].replace('  T', 0.0001, inplace=True)

    weather.at[2410, 'StnPressure'] = 29.34
    weather.at[2411, 'StnPressure'] = 29.34
    weather.at[2943, 'Depart'] = -5
    fillMissing(weather, 'Heat')
    fillMissing(weather, 'Cool')
    fillMissing(weather, 'WetBulb')
    fillMissing(weather, 'PrecipTotal')
    fillMissing(weather, 'StnPressure')
    fillMissing(weather, 'Tavg')
    fillMissing(weather, 'AvgSpeed')
    fillMissing(weather, 'Depart')

    # Only use data from station #1
    weather = weather.loc[weather['Station'] == 1]
    weather['Tavg'] = pd.to_numeric(weather['Tavg'])
    weather['AvgSpeed'] = pd.to_numeric(weather['AvgSpeed'])
    weather['PrecipTotal'] = pd.to_numeric(weather['PrecipTotal'])
    weather['StnPressure'] = pd.to_numeric(weather['StnPressure'])
    weather['Depart'] = pd.to_numeric(weather['Depart'])
    weather['Cool'] = pd.to_numeric(weather['Cool'])
    weather['Heat'] = pd.to_numeric(weather['Heat'])
    weather['WetBulb'] = pd.to_numeric(weather['WetBulb'])

    weather['DewDiff'] = weather['Tavg'] - weather['DewPoint']

    # Set date column as index
    weather.set_index('Date', inplace=True)


    return weather


def loadTraps(filename):
    # Load the data. We only want certain features of each.

    trap_features = ['Date', 'Species', 'Trap', 'Latitude',
                     'Longitude', 'NumMosquitos', 'WnvPresent']

    traps = pd.read_csv(filename)[trap_features]
    carriers = ('CULEX PIPIENS', 'CULEX RESTUANS', 'CULEX PIPIENS/RESTUANS')

    vector_weights = {'CULEX PIPIENS': .436, #.33, #.436,
                      'CULEX RESTUANS': .089, #.17, # .089,
                      'CULEX PIPIENS/RESTUANS': .475, #.49
                      'CULEX ERRATICUS': 0,
                      'CULEX TERRITANS': 0,
                      'CULEX SALINARIUS': 0,
                      'CULEX TARSALIS': 0}

    vector = [1 if species in carriers else 0 for species in traps['Species']]
    traps['vector'] = vector

    # Weighted vectors by species
    weights = [vec*vector_weights[species] for vec, species in
               zip(traps['vector'], traps['Species'])]
    traps['vector_weighted'] = weights


    # Set date column as index


    # Aggregate the column leakage
    grouped = traps.groupby(by=['Date', 'Latitude', 'Longitude', 'Species'])
    groupedsums = grouped.sum()

    # Remove duplicate columns
    nodupes = traps.drop_duplicates(subset=['Date',
                                            'Latitude',
                                            'Longitude',
                                            'Species'])

    print nodupes['NumMosquitos'].values.shape
    print groupedsums['NumMosquitos'].values.shape
    # Then sub in the added up row values
    nodupes['NumMosquitos'] = groupedsums['NumMosquitos'].values
    nodupes['WnvPresent'] = groupedsums['WnvPresent'].values

    # Cap the WnvPresent values at 1
    nodupes['WnvPresent'][nodupes['WnvPresent'] > 1] = 1

    species_encoded = pd.get_dummies(nodupes['Species'], prefix_sep='')
    traps = pd.concat([nodupes, species_encoded], axis=1)

    traps.drop(columns='Species', inplace=True)
    traps['Date'] = traps['Date'].apply(dateConverter)
    traps.set_index('Date', inplace=True)



    return traps


def loadTest(filename):
    test = pd.read_csv(filename)
    test.drop(columns=['Id', 'Address', 'Block', 'Street', 'Trap',
                       'AddressNumberAndStreet', 'AddressAccuracy'], inplace=True)
    species_encoded = pd.get_dummies(test['Species'], prefix_sep='')
    test = pd.concat([test, species_encoded], axis=1)

    return test


traps = loadTraps('../input/train.csv')
weather = loadWeather('../input/weather.csv')
test = loadTest('../input/test.csv')

# Mosquito population growth is affected by temperatures before the day
# they are captured. Let's do a rolling mean taking into account the previous
# X days (start with 10) and see if we can improve.


# Test different lags
# lags = range(1, 25)
# corr1 = []
# corr2 = []
# for lag in lags:
    # weather_traps['PrecipLag'] = pd.to_numeric(weather['PrecipTotal'].shift(periods=lag))
    # corr1.append(weather_traps[['PrecipLag', 'NumMosquitos']].corr().values[0, 1])
    #
    # weather_traps['Tavg-RollingMean'] = weather_traps['Tavg'].rolling(lag).mean()
    # corr1.append(weather_traps[['Tavg-RollingMean', 'NumMosquitos']].corr().values[0, 1])
#
# fig, ax = plt.subplots()
# ax.plot(lags, corr1)
# ax.set_ylabel('Correlation')
# ax.set_xlabel('Avg Window')
# fig.savefig('preciplag.png')

# Create new features
weather['WetBulb-depression'] = weather['Tavg'] - weather['WetBulb']
weather['PrecipCum-lag'] = weather['PrecipTotal'].rolling(3).sum().shift(periods=10)
weather['PrecipCum'] = weather['PrecipTotal'].rolling(3).sum()
weather['PrecipLag'] = pd.to_numeric(weather['PrecipTotal'].shift(periods=10))
weather['Tavg-RollingMean'] = weather['Tavg'].rolling(25).mean()
weather['WetBulb-RollingMean'] = weather['WetBulb'].rolling(25).mean()
weather['DewPoint-RollingMean'] = weather['DewPoint'].rolling(25).mean()
weather['WetBulb-depression-RollingMean'] = weather['WetBulb-depression'].rolling(25).mean()

weather['year'] = [val.year for val in weather.index]
weather['month'] = [val.month for val in weather.index]

weather_traps = traps.join(weather, how='left')  # join weather by date
weather_tests = test.join(weather, how='left')  # join weather by date

speciesnames = [u'CULEX ERRATICUS', u'CULEX PIPIENS', u'CULEX PIPIENS/RESTUANS',
                u'CULEX RESTUANS', u'CULEX SALINARIUS', u'CULEX TARSALIS',
                u'CULEX TERRITANS']

# nums = weather_traps['NumMosquitos'].values
# species = weather_traps[speciesnames].values
# numspecies = np.array([nums*species[:, n] for n in range(7)])
#
# weather_traps[speciesnames] = numspecies.transpose()
#
# with open('datatest.p', 'wb') as f:
#     pickle.dump(weather_tests, f)


with open('data.p', 'wb') as f:
    pickle.dump(weather_traps, f)

# Package imports
import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set matplot graph style
style.use('ggplot')

# Extract data from CSV dataset and set Date column as an index
df = pd.read_csv("datasets/2/GOOG.csv")
df.set_index('Date', inplace=True)

# Generate and initialize HL_PCT and PCT_change variables
df['HL_PCT'] = (df['High']-df['Low'])/df['Low']
df['PCT_change'] = (df['Adj Close']-df['Open'])/df['Open']
df = df[['HL_PCT', 'PCT_change', 'Adj Close', 'Volume']]

# Process data for forecast results prediction
forecast_col = 'Adj Close'
forecast_out = math.ceil(0.01 * len(df))   # a part of data that we will be forecasting

df.fillna(-99999, inplace=True)
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = X[:-forecast_out]
X_predict = X[-forecast_out:]
y = np.array(df['label'])
y = y[:-forecast_out]

# Train and test ML model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

# Calculate accuracy of the ML model
accuracy = clf.score(X_test, y_test)

# Predict forecast results from trained model
forecast_value = clf.predict(X_predict)

# Convert forecast data using numpy and set values as NaN
df['Forecast'] = np.nan

# Find out the last Adj Close data date
last_date_ = df.iloc[-1].name
last_date = datetime.datetime.strptime(last_date_, "%Y-%m-%d")
last_unix = last_date.timestamp()

# Add more days into the last date for forecasting
one_day = 24 * 60 * 60  # seconds in a day
next_unix = last_unix + one_day

for i in forecast_value:
    Date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[Date] = [np.nan for all in range(len(df.columns) - 1)] + [i]

# Show matplot graph for Adj Close and Forecast results
df['Adj Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Dates')
plt.ylabel('Price')
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.show()

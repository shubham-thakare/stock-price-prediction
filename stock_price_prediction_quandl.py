# Package imports
import datetime

import math
import matplotlib.pyplot as plt
import numpy as np
import quandl
from matplotlib import style
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression

from print_and_sleep import print_and_sleep

# Set matplot graph style
style.use('ggplot')

# Fetch GOOGLE stock price data from Quandl.com
print("Fetching GOOGLE stock price data from Quandl.com...")
df = quandl.get("WIKI/GOOGL")
print("Data fetching completed!")

# Initialize HL_PCT and PCT_change variables
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Process data for forecast results prediction
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

# Train and test ML model
print_and_sleep("\nTraining ML model...")
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
print("Trained!")

# Calculate accuracy of the ML model
print_and_sleep("\nCalculating trained model accuracy...")
confidence = clf.score(X_test, y_test)
print('Model Accuracy >>>', "{:.2f}".format(confidence * 100))

# Predict forecast results from trained model
print_and_sleep("\nPredicting forecast results...")
forecast_set = clf.predict(X_lately)

# Convert forecast data using numpy and set values as NaN
df['Forecast'] = np.nan

# Find out the last Adj Close data date
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()

# Add more days into the last date for forecasting
one_day = 24 * 60 * 60  # seconds in a day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

# Set graph window full screen
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")

# Show matplot graph for Adj Close and Forecast results
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

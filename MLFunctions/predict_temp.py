import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection

# Import data
data = pd.read_csv("/workspaces/WeatherSuggestion/MLFunctions/weatherdata/72429793812.csv")

# Modify data
# Separate dates first
data['DATE'] = pd.to_datetime(data['DATE'])
data['YEAR'] = data['DATE'].dt.year
data['MONTH'] = data['DATE'].dt.month
data['DAY'] = data['DATE'].dt.day

# Remove unneeded columns
removedColumns = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME", "TEMP_ATTRIBUTES", "DEWP_ATTRIBUTES", "SLP_ATTRIBUTES", "STP_ATTRIBUTES", "VISIB_ATTRIBUTES", "WDSP_ATTRIBUTES", "MAX_ATTRIBUTES", "MIN_ATTRIBUTES", "PRCP_ATTRIBUTES", "SNDP", "FRSHTT"]
data.drop(labels=removedColumns, axis=1, inplace=True)

# Fix faulty STP column
data.loc[data['STP'] < 100, "STP"] = data['STP'] + 1000

# Fix missing data in GUST and PRCP
data.loc[data['PRCP'] == 99.99, "PRCP"] = 0
data.loc[data['GUST'] == 999.9, "GUST"] = 15

# Add hot and cold labels
mean_temp = data["TEMP"].mean()
std_temp = data["TEMP"].std()
data['ZSCORE_TEMP'] = (data['TEMP'] - mean_temp) / std_temp
data["LABEL"] = 0
data.loc[data['ZSCORE_TEMP'] > 1, 'LABEL'] = 1
data.loc[data['ZSCORE_TEMP'] < -1, 'LABEL'] = -1

print(data)

# Create train test split
X_set = data.drop("LABEL", axis=1)
Y_set = data["LABEL"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_set, Y_set, test_size=0.2)

# Build and run model
randomForest = sklearn.ensemble.RandomForestClassifier(max_depth=2, random_state=0)
randomForest.fit(X_train, y_train)
predictions = randomForest.predict(X_test)

# Analyze results
print(sklearn.metrics.accuracy_score(y_test, predictions))
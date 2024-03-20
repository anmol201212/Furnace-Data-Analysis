import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.models import load_model
from keras.callbacks import CSVLogger

df = pd.read_csv('FurnaceData.csv')

# Split the data into training and testing sets
X = df.drop('GasConsumption', axis=1)  # Features
y = df['GasConsumption']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model1 = LinearRegression()
model1.fit(X_train, y_train)


#decesion Tree
modelDecTree = DecisionTreeRegressor()
modelDecTree.fit(X_train, y_train)


#Random forest
ModelRandForest = RandomForestRegressor()
ModelRandForest.fit(X_train, y_train)


# Make predictions on the test set

y_pred = model1.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


dump(model1, 'LinearRgressionModel1.joblib')
dump(modelDecTree, 'DecisionTreeModel2.joblib')
dump(ModelRandForest, 'RandomForestModel3.joblib')


model1Loaded = load('LinearRgressionModel1.joblib')
Predata = pd.read_csv('Prediction.csv')
Pred = model1Loaded.predict(Predata)


print('-------------------------------Model1 Ready ------------------------------------------------------------')

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
dump(scaler, 'scaler.joblib')


Pred_sclaeddata = scaler.transform(Predata)
# Create a sequential model
model2 = Sequential()

# Add input layer and hidden layers
model2.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))

# Add output layer
model2.add(Dense(1, activation='linear'))

# Compile the model
model2.compile(optimizer='adam', loss='mean_squared_error')

log_filename = 'Deeplearningtraining_log.csv'
csv_logger = CSVLogger(log_filename)

# Train the model
model2.fit(X_train_scaled, y_train, epochs=150, batch_size=50, verbose=0, callbacks=[csv_logger])

# Evaluate the model
mse = model2.evaluate(X_test_scaled, y_test)
print("Mean Squared Error:", mse)

# Make predictions
y_pred = model2.predict(X_test_scaled)

# Print example predictions
for i in range(5):
    print("Actual:", y_test.iloc[i], "\tPredicted:", y_pred[i][0])

model2.save("DeeplearningModel4.h5")
#model = load_model("my_model.h5")

PredMdl2 = model2.predict(Pred_sclaeddata)
PredMdlDctres = modelDecTree.predict(Predata)
PredMdl2RandomForest = ModelRandForest.predict(Predata)
print(PredMdl2)
# print(Pred)
# print(PredMdlDctres)
# print(PredMdl2RandomForest)

print('-----------------------------------------------')


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
import numpy as np
import matplotlib.pyplot as plt



def PredLinearRegressionModel(df):
    #df = pd.read_csv(CSVPth)
    model = load('Models/LinearRgressionModel1.joblib')
    Prediction = model.predict(df)
    Prediction = np.round(Prediction, 2)
    return Prediction

def PredDecisionTreeModel(df):
    model = load('Models/DecisionTreeModel2.joblib')
    Prediction = model.predict(df)
    Prediction = np.round(Prediction, 2)
    return Prediction

def PredRandomForestModel(df):
    model = load('Models/RandomForestModel3.joblib')
    Prediction = model.predict(df)
    Prediction = np.round(Prediction, 2)
    return Prediction


def PredDeeplearningModel(df):
    scaler = load('Models/scaler.joblib')
    Model = load_model("Models/DeeplearningModel4.h5")
    Pred_sclaeddata = scaler.transform(df)
    PredDeeplearning = Model.predict(Pred_sclaeddata)
    PredDeeplearning = PredDeeplearning.flatten()
    PredDeeplearning = np.round(PredDeeplearning, 2)
    return PredDeeplearning









# print('-----------------------------------------------')
# PredLinearRegre = np.round(PredLinearRegre, 2)
# print(PredLinearRegre)
# print('-----------------------------------------------')
# PredDecisionTree = np.round(PredDecisionTree, 2)
# print(PredDecisionTree)
# print('-----------------------------------------------')
# PredRandomForest = np.round(PredRandomForest, 2)
# print(PredRandomForest)
# print('-----------------------------------------------')

# PredDeeplearning = PredDeeplearning.flatten()
# PredDeeplearning = np.round(PredDeeplearning, 2)
# print(PredDeeplearning)

# plt.plot(PredLinearRegre, label='LinearRegression_Prediction 1')
# plt.plot(PredDecisionTree, label='DecisionTreeRegressor_Prediction')
# plt.plot(PredRandomForest, label='RandomForestRegressor_Prediction')
# plt.plot(PredDeeplearning, label='DeeplarningModel_Prediction')


# PredLinearRegre = LinearRegressionModel.predict(df)
# PredDecisionTree = DecisionTreeRegressorModel.predict(df)
# PredRandomForest = RandomForestRegressorModel.predict(df)
# PredDeeplearning = DeeplarningModel.predict(Pred_sclaeddata)

# # Title and labels
# plt.title('Predicted Values')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')

# # Legend
# plt.legend()

# # Displaying the chart
# plt.show()



# print('-----------------------------------------------')


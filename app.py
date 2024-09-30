from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv("rainfall_data.csv", encoding="ISO-8859-1")

groups = dataset.groupby('SUBDIVISION')[['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','NOV','DEC']]
data=groups.get_group(('COASTAL ANDHRA PRADESH'))

data=data.melt(['YEAR']).reset_index()

df = data[['YEAR', 'variable', 'value']].reset_index().sort_values(by=['YEAR', 'index'])

df.columns = ['INDEX', 'YEAR', 'Month', 'avg_rainfall']

df['Month'] = df['Month'].map({'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6, 'JUL':7, 'AUG':8, 'SEP':9,
                               'OCT':10, 'NOV':11, 'DEC':12})
df['Date'] = pd.to_datetime(df.assign(Day=1).loc[:,['YEAR','Month','Day']])

cols = ['avg_rainfall']
dataset = df[cols]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(dataset[['avg_rainfall']])

# Split the data into training and testing sets
TRAIN_SIZE = 0.60
train_size = int(len(data_scaled) * TRAIN_SIZE)
test_size = len(data_scaled) - train_size
train, test = data_scaled[0:train_size, :], data_scaled[train_size:len(data_scaled), :]

# Function to create dataset
def create_dataset(dataset, window_size=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return np.array(data_X), np.array(data_Y)

window_size = 1
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)

# Reshape input data for LSTM
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# Define and train the LSTM model
def create_model(train_X, train_Y, window_size=1):
    model = Sequential()
    model.add(LSTM(2000, activation='tanh', recurrent_activation='hard_sigmoid', input_shape=(1, window_size)))
    model.add(Dropout(0.2))
    model.add(Dense(500))
    model.add(Dropout(0.4))
    model.add(Dense(500))
    model.add(Dropout(0.4))
    model.add(Dense(400))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_X, train_Y, epochs=30, batch_size=10)
    return model

model = create_model(train_X, train_Y, window_size)

# Function to predict and score
def predict_and_score(model, X, Y, scaler):
    pred = scaler.inverse_transform(model.predict(X))
    return pred[-1][0]

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    month = int(request.form['month'])
    data = np.array([[year + month / 12]])
    scaled_data = scaler.transform(data)
    scaled_data = np.reshape(scaled_data, (scaled_data.shape[0], 1, scaled_data.shape[1]))
    prediction = predict_and_score(model, scaled_data, 0, scaler)  # 0 is a placeholder for Y in predict_and_score function
    prediction = float(prediction/10)  # Convert prediction to Python float
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
import tensorflow as tf
import numpy as np
import math
from tensorflow import keras 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def time_series_split(dataset: pd.DataFrame, time_step=1) -> tuple[np.array, np.array]:
    x_data, y_data = [], []
    for i in range(len(dataset) - time_step-1):
        a = dataset[i:(i+time_step),0]
        x_data.append(a)
        y_data.append(dataset[i+time_step, 0])
    return np.array(x_data), np.array(y_data)

def sample_output(timestep, model, scaler, df_scaled, days) -> list[int]:
    out = []
    x_input = df_scaled[-timestep:].reshape(1,timestep,1)
    temp_input = x_input.flatten().tolist()
    print(x_input.shape)
    y_output = model.predict(x_input)
    print(f'Day 1 output: {y_output}')
    temp_input.extend(y_output[0].tolist())
    out.extend(y_output.tolist())

    for i in range(2,days+1):
        x_input = np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape(1,timestep,1)
        print(x_input.shape)
        y_output = model.predict(x_input)
        print(f'Day {i} output: {y_output}')
        temp_input.extend(y_output[0].tolist())
        temp_input=temp_input[1:]
        out.extend(y_output.tolist())
    out = np.array(out).flatten()
    out = [int(scaler.inverse_transform(x.reshape(1,-1)).flatten()) for x in out]
    print(f'output = {out}')
    return out

def plot_results(stock_name: str, timestep, scaler, df_scaled, train_predict, test_predict, out) -> None:
    timestep=100
    trainPredictPlot = np.empty_like(df_scaled)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[timestep:len(train_predict)+timestep, :] = train_predict
    testPredictPlot = np.empty_like(df_scaled)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(timestep*2)+1:len(df_scaled)-1, :] = test_predict 
    actualPredictPlot = np.empty(len(df_scaled)+30)
    actualPredictPlot[:] = np.nan
    actualPredictPlot[len(df_scaled):len(actualPredictPlot)] = out
    plt.plot(scaler.inverse_transform(df_scaled), 'g', label='Actual Price')
    plt.plot(trainPredictPlot, 'y', label='Prediction (training set')
    plt.plot(testPredictPlot, 'r', label='Prediction (test set)')
    plt.plot(actualPredictPlot, 'b', label="Future Prediction")
    plt.title(f"{stock_name.upper()} Stock Price")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.legend(loc='upper left')
    plt.show()

def build_neural_net(stock_name: str) -> None:
    STEP = 100 # time step; 100 days used for a prediction 
    df = pd.read_csv(f'data/{stock_name}.csv')
    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df['Close'].values.reshape(-1,1))

    train, test = train_test_split(df_scaled, test_size=0.2, shuffle=False)
    X_train, y_train = time_series_split(train, STEP)
    X_test, y_test = time_series_split(test, STEP)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = tf.keras.Sequential([
        keras.layers.LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        keras.layers.Dropout(0.5),
        keras.layers.LSTM(units=100, return_sequences=True),
        keras.layers.Dropout(0.5),
        keras.layers.LSTM(units=100),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=2048, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    print(math.sqrt(mean_squared_error(y_train, train_predict)))
    print(math.sqrt(mean_squared_error(y_test, test_predict)))

    out = sample_output(STEP, model, scaler, df_scaled, 30)
    plot_results(stock_name, STEP, scaler, df_scaled, train_predict, test_predict, out)

    model_json = model.to_json()
    with open(f"models/{stock_name}.json", "w") as json_file: json_file.write(model_json)
    model.save_weights(f"models/{stock_name}.h5")
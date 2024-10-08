from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import os

app = Flask(__name__)

# Check if the model exists; if not, train and save it
model_file = 'stock_model.h5'

def create_model():
    model = Sequential()
    model.add(LSTM(units=20, return_sequences=True, input_shape=(60, 4)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=20, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def fetch_and_predict(symbol):
    today = pd.Timestamp("today").strftime('%Y-%m-%d')
    data = yf.download(symbol, start="2015-01-01", end=today)

    # Feature Engineering
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data.dropna(inplace=True)

    # Preparing the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'SMA_50', 'SMA_200', 'RSI']])

    # Splitting the data into train and test sets
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    # Creating sequences for LSTM model
    def create_sequences(data, look_back=60):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i-look_back:i, :])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    look_back = 60
    X_train, y_train = create_sequences(train_data, look_back)

    # Load or create the model
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = create_model()
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, callbacks=[early_stop])
        model.save(model_file)

    # Making predictions
    latest_data = scaled_data[-look_back:]
    latest_prediction = model.predict(latest_data.reshape(1, look_back, 4))
    predicted_price = scaler.inverse_transform(np.concatenate((latest_prediction, np.zeros((1, 3))), axis=1))[:, 0][0]

    # Get the latest closing price
    closing_price = data['Close'].iloc[-1]

    # Determine the trend
    trend = "Upward" if predicted_price > closing_price else "Downward"

    return round(closing_price, 2), round(predicted_price, 2), trend

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    closing_price, predicted_price, trend = fetch_and_predict(symbol)

    return render_template('index.html', 
                           closing_price=closing_price, 
                           predicted_price=predicted_price, 
                           trend=trend)

if __name__ == '__main__':
    app.run(debug=True)

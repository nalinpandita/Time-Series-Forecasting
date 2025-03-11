from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.api import VAR
from pykalman import KalmanFilter

app = Flask(__name__, static_url_path='/static')

STATIC_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        if df.isnull().values.any():
            df.fillna(method='ffill', inplace=True)

        # Apply Kalman filter to 'Close' prices
        df['Close'] = apply_kalman_filter(df['Close'].values)

        return df

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File '{file_path}' not found. Check if the file exists in the correct location.")
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def apply_kalman_filter(data):
    kf = KalmanFilter(transition_matrices=[1],
                    observation_matrices=[1],
                    initial_state_mean=data[0],
                    initial_state_covariance=1,
                    observation_covariance=1,
                    transition_covariance=0.01)

    filtered_state_means, _ = kf.filter(data)
    return filtered_state_means.flatten()


def create_lstm_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def suggest_action(forecasted_values, current_price):
    if current_price > forecasted_values[-1]:
        return "sell"
    elif current_price < forecasted_values[-2]:
        return "buy"
    elif all(forecasted_values[i] <= forecasted_values[i + 1] for i in range(len(forecasted_values) - 1)):
        return "buy"
    else:
        return "sell"

def forecast_with_var(df, investment_days):
    try:
        model = VAR(df)
        model_fit = model.fit()
        forecast = model_fit.forecast(model_fit.endog[-model_fit.k_ar:], steps=investment_days)
        forecast_df = pd.DataFrame(forecast, index=pd.date_range(start=df.index[-1], periods=investment_days+1, freq='B')[1:], columns=df.columns)
        return forecast_df['Close'].values, forecast_df['Volume'].values
    except Exception as e:
        raise ValueError(f"Error in VAR forecasting: {e}")

def main(file_path, investment_days, num_stocks):
    try:
        df = load_data(file_path)
        ts_close = df[['Close']].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        ts_close_scaled = scaler.fit_transform(ts_close)

        time_step = 60
        X, Y = create_lstm_dataset(ts_close_scaled, time_step)

        training_size = int(len(X) * 0.8)
        X_train, X_test = X[:training_size], X[training_size:]
        Y_train, Y_test = Y[:training_size], Y[training_size:]

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = build_lstm_model((X_train.shape[1], 1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, batch_size=64, verbose=1, callbacks=[early_stopping])

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        lstm_forecasted_values = []
        x_input = X_test[-1]
        for _ in range(investment_days):
            x_input = x_input.reshape((1, time_step, 1))
            yhat = model.predict(x_input, verbose=0)
            lstm_forecasted_values.append(yhat[0][0])
            x_input = np.append(x_input[:, 1:, :], yhat.reshape((1, 1, 1)), axis=1)

        lstm_forecasted_values = scaler.inverse_transform(np.array(lstm_forecasted_values).reshape(-1, 1))

        var_forecasted_close, var_forecasted_volume = forecast_with_var(df[['Close', 'Volume']], investment_days)

        combined_forecasted_values = (lstm_forecasted_values.flatten() + var_forecasted_close) / 2.0

        current_price = df['Close'].iloc[-1]

        suggestion = suggest_action(combined_forecasted_values, current_price)

        investment = num_stocks * current_price
        forecasted_last_price = combined_forecasted_values[-1]

        profit_per_stock = abs(current_price - forecasted_last_price)
        returns = profit_per_stock * num_stocks + investment

        return current_price, combined_forecasted_values, suggestion, investment, returns

    except FileNotFoundError as e:
        app.logger.error(f"File not found: {e}")
        raise FileNotFoundError(f"File not found: {e}")

    except ValueError as e:
        app.logger.error(f"Value error: {e}")
        raise ValueError(f"Value error: {e}")

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        raise Exception(f"An error occurred: {e}")


@app.route('/')
def index():
    datasets = [f for f in os.listdir(STATIC_FOLDER) if f.endswith('.csv')]
    return render_template('index.html', datasets=datasets)

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        investment_days = int(data['investment_days'])
        num_stocks = int(data['num_stocks'])
        dataset = data['dataset']

        file_path = os.path.join(STATIC_FOLDER, dataset)

        current_price, combined_forecasted_values, suggestion, investment, returns = main(file_path, investment_days, num_stocks)

        return jsonify({
            'current_price': round(current_price, 2),
            'forecasted_values': [round(val, 2) for val in combined_forecasted_values],
            'suggestion': suggestion,
            'investment': round(investment, 2),
            'profit': round(returns, 2)
        })

    except FileNotFoundError as e:
        app.logger.error(f"File not found: {e}")
        return jsonify({'error': str(e)}), 400

    except ValueError as e:
        app.logger.error(f"Value error: {e}")
        return jsonify({'error': str(e)}), 500

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

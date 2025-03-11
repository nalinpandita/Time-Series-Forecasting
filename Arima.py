from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.impute import SimpleImputer

app = Flask(__name__, static_url_path='/static')

STATIC_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

def load_data(file_path):
    try:
        abs_file_path = os.path.join(STATIC_FOLDER, file_path)
        df = pd.read_csv(abs_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.asfreq('D')

        if df.isnull().values.any():
            imputer = SimpleImputer(strategy='mean')
            df[['Open', 'Close', 'High', 'Low']] = imputer.fit_transform(df[['Open', 'Close', 'High', 'Low']])
        
        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File '{abs_file_path}' not found. Check if the file exists in the correct location.")

def check_seasonality(ts):
    if len(ts) < 730:
        return None
    else:
        decomposition = seasonal_decompose(ts, model='additive', period=365)
        return decomposition

def remove_seasonality(ts, decomposition):
    if decomposition:
        detrended = ts - decomposition.seasonal
    else:
        detrended = ts
    return detrended

def impute_missing_values(ts_detrended, trend):
    try:
        if ts_detrended.isnull().any():
            imputer = SimpleImputer(strategy='mean')
            ts_detrended_filled = pd.Series(imputer.fit_transform(ts_detrended.values.reshape(-1, 1)).flatten(), index=ts_detrended.index)
            ts_imputed = ts_detrended_filled.fillna(trend.mean())
            return ts_imputed
        else:
            return ts_detrended
    except Exception as e:
        raise ValueError("Error in impute_missing_values: " + str(e))

def check_stationarity(ts):
    result = adfuller(ts)
    return result[1] <= 0.05

def make_stationary(ts):
    diff_ts = ts.diff().dropna()
    is_stationary = check_stationarity(diff_ts)
    d = 1
    while not is_stationary:
        diff_ts = diff_ts.diff().dropna()
        is_stationary = check_stationarity(diff_ts)
        d += 1
    return diff_ts, d

def fit_arima_model(ts, d, max_p=5, max_q=5):
    p = q = range(0, max_p + 1)
    pdq = [(x, d, y) for x in p for y in q]
    
    best_aic = np.inf
    best_bic = np.inf
    best_order = None
    best_mdl = None
    
    for param in pdq:
        try:
            tmp_mdl = SARIMAX(ts, order=param, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            tmp_aic = tmp_mdl.aic
            tmp_bic = tmp_mdl.bic
            if tmp_aic < best_aic and tmp_bic < best_bic:
                best_aic = tmp_aic
                best_bic = tmp_bic
                best_order = param
                best_mdl = tmp_mdl
        except: continue

    return best_mdl, best_order


def forecast_future_values(ts, model_fit, seasonal_component, periods):
    forecast = model_fit.get_forecast(steps=periods).predicted_mean
    
    # Reverting differencing if applied
    if isinstance(ts, pd.DataFrame):
        forecast = ts.iloc[-1] + forecast.cumsum()
    else:
        forecast = ts[-1] + forecast.cumsum()

    if seasonal_component is not None:
        seasonal_forecast = np.tile(seasonal_component[-periods:], periods // len(seasonal_component) + 1)[:periods]
        forecast += seasonal_forecast

    return pd.Series(forecast)

def calculate_cumulative_return(initial_investment, forecasted_values):
    cumulative_return = initial_investment * (1 + forecasted_values.pct_change().cumsum())
    return cumulative_return.iloc[-1]

def suggest_action(forecasted_values, current_price):
    if all(forecasted_values.iloc[i] <= forecasted_values.iloc[i + 1] for i in range(len(forecasted_values) - 1)):
        if forecasted_values.iloc[-1] > current_price:
            return "Buy"
        else:
            return "Sell"
    else:
        if forecasted_values.iloc[-1] < current_price:
            return "Sell"
        else:
            return "Buy"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        investment_days = int(data['investment_days'])
        num_stocks = int(data['num_stocks'])

        df = load_data('TATAMOTORS.csv')
        ts_close = df['Close']

        decomposition_close = check_seasonality(ts_close)
        if decomposition_close:
            ts_close_detrended = remove_seasonality(ts_close, decomposition_close)
            trend_close = decomposition_close.trend
        else:
            ts_close_detrended = ts_close
            trend_close = ts_close

        ts_close_imputed = impute_missing_values(ts_close_detrended, trend_close)

        # Choose best ARIMA model order and seasonal order
        model_fit_close, best_order = fit_arima_model(ts_close_imputed, d=0)

        forecasted_close = forecast_future_values(ts_close_imputed, model_fit_close, decomposition_close.seasonal.values if decomposition_close else None, investment_days)

        initial_investment = 1000
        cumulative_return_close = calculate_cumulative_return(initial_investment, forecasted_close)

        current_price = df['Close'].iloc[-1]
        investment = num_stocks * current_price

        suggestion = suggest_action(forecasted_close, current_price)

        forecasted_last_price = round(forecasted_close.iloc[-1], 2)

        if suggestion == "Buy":
            profit_per_stock = abs(current_price - forecasted_last_price)
            returns = profit_per_stock * num_stocks + investment
        else:
            profit_per_stock = abs(current_price - forecasted_last_price)
            returns = profit_per_stock * num_stocks + investment

        cumulative_return_after_investment = initial_investment + returns

        return jsonify({
            'current_price': current_price,
            'forecasted_values': [round(val, 2) for val in forecasted_close.tolist()],
            'suggestion': suggestion,
            'investment': investment,
            'profit': round(returns, 2),
            'best_order': best_order
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
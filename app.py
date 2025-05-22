from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from joblib import load
import datetime
import pickle
import json
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from pymongo import MongoClient


# MongoDB connection setup
client = MongoClient('mongodb://localhost:27017/')
db = client['energy_predictions']
collection = db['predictions']

app = Flask(__name__)

# Load the SVR model and scalers
svr_model = load('models/svr_model.joblib')
scaler_X = load('models/scaler_X.joblib')
scaler_y = load('models/scaler_y.joblib')

# Load the Prophet model
with open('models/Prophet.pkl', 'rb') as f:
    prophet_model = pickle.load(f)

# Load the Exponential Smoothing model
with open('models/ETS.pkl', 'rb') as f:
    ets_model = pickle.load(f)

# Load the ARIMA model
with open('models/arima_model.pkl', 'rb') as f:
    arima_model = pickle.load(f)


# Load the SARIMA model
with open('models/SARIMA.pkl', 'rb') as f:
    sarima_model = pickle.load(f)


# Load the MinMaxScaler used for ANN
ann_scaler = MinMaxScaler()
data = pd.read_csv('dataset.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)
data_scaled = ann_scaler.fit_transform(data)

@app.route('/')
def home():
    return render_template('App.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/dashboard_data', methods=['POST'])
def dashboard_data():
    try:
        num_periods = int(request.form['num_periods'])
        selected_models = json.loads(request.form['models'])

        results = {'dates': pd.date_range(start=data.index.max(), periods=num_periods + 1, freq='D')[1:].strftime('%Y-%m-%d').tolist()}
        predictions = {}

        if 'arima' in selected_models:
            predictions['arima'] = arima_model.forecast(steps=num_periods).tolist()

        if 'sarima' in selected_models:
            predictions['sarima'] = sarima_model.get_forecast(steps=num_periods).predicted_mean.tolist()

        if 'ets' in selected_models:
            predictions['ets'] = ets_model.forecast(steps=num_periods).tolist()

        if 'prophet' in selected_models:
            future = prophet_model.make_future_dataframe(periods=num_periods, freq='D')
            prophet_forecasts = prophet_model.predict(future)
            predictions['prophet'] = prophet_forecasts.tail(num_periods)['yhat'].tolist()

        if 'svr' in selected_models:
            last_date_str = '2001-01-02'  # Example fixed date
            last_date = datetime.datetime.strptime(last_date_str, '%Y-%m-%d')
            future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, num_periods + 1)]
            future_dates_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
            future_dates_scaled = scaler_X.transform(future_dates_ordinal)
            future_predictions_scaled = svr_model.predict(future_dates_scaled)
            future_predictions = scaler_y.inverse_transform(future_predictions_scaled.reshape(-1, 1))
            predictions['svr'] = future_predictions.flatten().tolist()

        # Store predictions in MongoDB
        for model_name, model_predictions in predictions.items():
            prediction_data = {
                'model': model_name,
                'predictions': model_predictions,
                'dates': results['dates']
            }
            collection.insert_one(prediction_data)

        # Combine predictions with results for response
        results.update(predictions)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_svr', methods=['POST'])
def predict_svr():
    try:
        # Get number of periods to predict from the form
        num_days = int(request.form['num_periods'])

        # Predict for future steps: if training was on 0 to N-1, next is N to N+num_days-1
        start_index = len(data)
        future_steps = np.arange(start_index, start_index + num_days).reshape(-1, 1)

        # Scale future steps using the same scaler as in training
        future_steps_scaled = scaler_X.transform(future_steps)

        # Predict using SVR
        predictions_scaled = svr_model.predict(future_steps_scaled)
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))

        # Use relative labels like "Day 1", "Day 2", etc.
        results = {
            'dates': [f'Day {i+1}' for i in range(num_days)],
            'predictions': predictions.flatten().tolist()
        }

        return jsonify(results)

    except Exception as e:
        print(f"Error in SVR prediction: {e}")
        return jsonify({'error': str(e)}), 400



@app.route('/predict_prophet', methods=['POST'])
def predict_prophet():
    try:
        # Get the number of periods for prediction from the form
        num_periods = int(request.form['num_periods'])

        # Create future dates to forecast
        future = prophet_model.make_future_dataframe(periods=num_periods, freq='D')

        # Predict future values
        forecast = prophet_model.predict(future)

        # Get the last `num_periods` predictions
        forecast_tail = forecast.tail(num_periods)

        # Prepare the response
        results = {
            'dates': forecast_tail['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'predictions': forecast_tail['yhat'].tolist(),
            'lower_bounds': forecast_tail['yhat_lower'].tolist(),
            'upper_bounds': forecast_tail['yhat_upper'].tolist()
        }

        return jsonify(results)

    except Exception as e:
        print(f"Error in Prophet prediction: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_ets', methods=['POST'])
def predict_ets():
    try:
        # Get the number of periods for prediction from the form
        num_periods = int(request.form['num_periods'])

        # Generate predictions using the loaded ETS model
        forecast = ets_model.forecast(steps=num_periods)

        # Prepare the response
        future_dates = pd.date_range(start=data.index.max(), periods=num_periods + 1, freq='D')[1:]
        results = {
            'dates': future_dates.strftime('%Y-%m-%d').tolist(),
            'predictions': forecast.tolist()
        }

        return jsonify(results)

    except Exception as e:
        print(f"Error in ETS prediction: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_arima', methods=['POST'])
def predict_arima():
    try:
        # Get the number of periods for prediction from the form
        num_periods = int(request.form['num_periods'])

        # Generate predictions using the loaded ARIMA model
        forecast = arima_model.forecast(steps=num_periods)

        # Prepare the response
        future_dates = pd.date_range(start=data.index.max(), periods=num_periods + 1, freq='D')[1:]
        results = {
            'dates': future_dates.strftime('%Y-%m-%d').tolist(),
            'predictions': forecast.tolist()
        }

        return jsonify(results)

    except Exception as e:
        print(f"Error in ARIMA prediction: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_sarima', methods=['POST'])
def predict_sarima():
    try:
        # Get the number of periods for prediction from the form
        num_periods = int(request.form['num_periods'])

        # Generate predictions using the loaded SARIMA model
        forecast = sarima_model.get_forecast(steps=num_periods)
        forecast_mean = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()

        # Prepare the response
        future_dates = pd.date_range(start=data.index.max(), periods=num_periods + 1, freq='D')[1:]
        results = {
            'dates': future_dates.strftime('%Y-%m-%d').tolist(),
            'predictions': forecast_mean.tolist(),
            'lower_bounds': confidence_intervals.iloc[:, 0].tolist(),
            'upper_bounds': confidence_intervals.iloc[:, 1].tolist()
        }

        return jsonify(results)

    except Exception as e:
        print(f"Error in SARIMA prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

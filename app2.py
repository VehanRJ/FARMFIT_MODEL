import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import tensorflow as tf
from flask import Flask, jsonify, request, render_template_string

# ------------------ GPU Memory Growth ------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ------------------ Load Models ------------------
scaler = joblib.load("LSTM_scaler.joblib")
model = load_model('latest_crop_TS_model.h5')
regressor = joblib.load('Regressor.joblib')
classifier = joblib.load('Classifier.joblib')

# ------------------ Helper Functions ------------------
def yes_or_no(x):
    return 'yes' if int(round(x)) == 1 else 'no'

def solution(df, last_row=None):
    dRisk = round(np.mean(df['disease_risk_score']) * 100, 2)
    dObserved = int(round(np.mean(df['disease_observed'])))
    pestAmt = round(np.mean(df['pesticide_level_ppm']), 4)

    maturity = np.mean(df['growth_stage_maturity'])
    tillering = np.mean(df['growth_stage_tillering'])
    grand_growth = np.mean(df['growth_stage_grand_growth'])
    emergence = np.mean(df['growth_stage_emergence'])

    my_dict = {
        'maturity': maturity,
        'tillering': tillering,
        'grand_growth': grand_growth,
        'emergence': emergence
    }
    value_to_find = np.max(list(my_dict.values()))
    keys = [k for k, v in my_dict.items() if v == value_to_find]
    growthStage = keys[0]

    res = {
        "disease_risk_percentage": dRisk,
        "disease_observed": yes_or_no(dObserved),
        "pesticide_amount_ppm": pestAmt,
        "growth_stage": growthStage
    }

    # Include temperature and timestamp from the last row if available
    if last_row is not None:
        res["temperature_C"] = float(last_row["air_temperature_C"])
        res["timestamp"] = str(last_row["timestamp"])
        res["relative_humidity_pct"] = float(last_row["relative_humidity_pct"])
        res["soil_moisture_pct"] = float(last_row["soil_moisture_pct"])
        res["soil_pH"] = float(last_row["soil_pH"])
        res["soil_EC_dS_m"] = float(last_row["soil_EC_dS_m"])
        res["nitrate_ppm"] = float(last_row["nitrate_ppm"])
        res["phosphorus_ppm"] = float(last_row["phosphorus_ppm"])
        res["potassium_ppm"] = float(last_row["potassium_ppm"])
        res["rainfall_mm"] = float(last_row["rainfall_mm"])
        res["irrigation_event"] = int(last_row["irrigation_event"])
        res["fertilizer_event"] = int(last_row["fertilizer_event"])
        res["pesticide_level_ppm"] = float(last_row["pesticide_level_ppm"])
        res["disease_risk_score"] = float(last_row["disease_risk_score"])
        res["disease_observed"] = int(last_row["disease_observed"])
        res["disease_risk_score"] = float(last_row["disease_risk_score"])
        res["growth_stage_emergence"] = int(last_row["growth_stage_emergence"])
        res["growth_stage_grand_growth"] = int(last_row["growth_stage_grand_growth"])
        res["growth_stage_maturity"] = int(last_row["growth_stage_maturity"])
        res["growth_stage_tillering"] = int(last_row["growth_stage_tillering"])



    return res

def forecast_from_dataset(df):
    df.drop(['timestamp'], axis=1, inplace=True, errors="ignore")
    train_scaled = scaler.transform(df)

    length = 30
    n_features = df.shape[1]
    first_eval_batch = np.array(train_scaled[-length:])
    current_batch = first_eval_batch.reshape(1, length, n_features)

    forcast = []
    for i in range(60):
        next_features = current_batch[:, -1, :].copy()
        pred = model.predict(current_batch, verbose=0)
        forcast.append(pred)
        current_batch = np.concatenate(
            [current_batch[:, 1:, :], next_features.reshape(1, 1, n_features)], axis=1
        )

    forcast = np.array(forcast).reshape(60, n_features)
    forcast = scaler.inverse_transform(forcast)
    forcast = pd.DataFrame(data=forcast, columns=df.columns)

    # Regressor prediction
    pred_main = regressor.predict(forcast.drop([
        'growth_stage_emergence', 'growth_stage_grand_growth',
        'growth_stage_maturity', 'growth_stage_tillering',
        'pesticide_level_ppm', 'disease_risk_score'
    ], axis=1))

    forcast['pesticide_level_ppm'] = pred_main[:, 0]
    forcast['disease_risk_score'] = pred_main[:, 1]

    # Classifier prediction
    pred_main1 = classifier.predict(forcast.drop([
        'disease_observed', 'growth_stage_emergence',
        'growth_stage_grand_growth', 'growth_stage_maturity',
        'growth_stage_tillering'
    ], axis=1))

    if pred_main1.ndim > 1 and pred_main1.shape[1] > 1:
        forcast['disease_observed'] = pred_main1[:, 1]
    else:
        forcast['disease_observed'] = pred_main1

    return forcast

# ------------------ Flask App ------------------
app = Flask(__name__)

# Example upload page
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Crop Prediction</title>
</head>
<body>
  <h1>Upload Dataset (CSV)</h1>
  <form action="/results" method="post" enctype="multipart/form-data">
    <input type="file" name="file" accept=".csv" required>
    <button type="submit">Submit</button>
  </form>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)

@app.route("/results", methods=["POST"])
def results():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    df = pd.read_csv(file)
    last_row = df.iloc[-1]  # Get last row for timestamp & temperature
    forcast = forecast_from_dataset(df)
    res = solution(forcast, last_row=last_row)
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=True)

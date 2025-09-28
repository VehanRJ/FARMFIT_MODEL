from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import models, layers, applications
from tensorflow.keras.preprocessing import image
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Image Model Setup
# -----------------------------
base = applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling='avg'
)

inputs = layers.Input(shape=(224, 224, 3))
x = base(inputs, training=False)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(6, activation='softmax')(x)
model_img = models.Model(inputs, outputs)
model_img.load_weights("dominator.h5")

label_dict = {
    0:'BacterialBlights',
    1:'Healthy',
    2:'Mosaic',
    3:'RedRot',
    4:'Rust',
    5:'Yellow'
}

solution_dict = {
    0: 'Use copper-based bactericides; avoid water stagnation and practice crop rotation.',
    1: 'No treatment needed; maintain proper irrigation and regular crop monitoring.',
    2: 'Control aphid vectors with neem oil or systemic insecticides; rogue out infected plants.',
    3: 'Remove and burn infected clumps; apply fungicides like carbendazim or triazoles.',
    4: 'Spray fungicides containing mancozeb or propiconazole; improve field drainage and spacing.',
    5: 'Apply balanced fertilizers with adequate nitrogen; foliar spray with zinc or iron if deficiency symptoms persist.'
}

# -----------------------------
# Dataset Model Setup
# -----------------------------
scaler = joblib.load("LSTM_scaler.joblib")
model_ts = tf.keras.models.load_model('latest_crop_TS_model.h5')
regressor = joblib.load('Regressor.joblib')
classifier = joblib.load('Classifier.joblib')

# -----------------------------
# Helper Functions
# -----------------------------
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
        pred = model_ts.predict(current_batch, verbose=0)
        forcast.append(pred)
        current_batch = np.concatenate(
            [current_batch[:, 1:, :], next_features.reshape(1, 1, n_features)], axis=1
        )

    forcast = np.array(forcast).reshape(60, n_features)
    forcast = scaler.inverse_transform(forcast)
    forcast = pd.DataFrame(data=forcast, columns=df.columns)

    pred_main = regressor.predict(forcast.drop([
        'growth_stage_emergence', 'growth_stage_grand_growth',
        'growth_stage_maturity', 'growth_stage_tillering',
        'pesticide_level_ppm', 'disease_risk_score'
    ], axis=1))

    forcast['pesticide_level_ppm'] = pred_main[:, 0]
    forcast['disease_risk_score'] = pred_main[:, 1]

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

# -----------------------------
# Example Upload Page
# -----------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Crop Analysis</title>
</head>
<body>
<h1>Upload Image & CSV Dataset</h1>
<form action="/analyze_form" method="post" enctype="multipart/form-data">
    <label>Leaf Image (JPG/PNG):</label>
    <input type="file" name="file_img" accept="image/*" required><br><br>
    <label>Dataset CSV:</label>
    <input type="file" name="file_csv" accept=".csv" required><br><br>
    <button type="submit">Analyze</button>
</form>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

# -----------------------------
# Form Upload Route
# -----------------------------
from fastapi.responses import JSONResponse

@app.post("/analyze_form")
async def analyze_form(file_img: UploadFile = File(...), file_csv: UploadFile = File(...)):
    try:
        # ---------- IMAGE ----------
        img = Image.open(file_img.file).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        pred = model_img.predict(img_array)
        top_idx = pred.argsort(axis=1)[0, -1]
        label = label_dict[top_idx]
        solution_text = solution_dict[top_idx]
        confidence = f"{np.max(pred)*100:.2f}"

        image_result = {
            "prediction": label,
            "solution": solution_text,
            "confidence": confidence
        }

        # ---------- DATASET ----------
        df = pd.read_csv(file_csv.file)
        last_row = df.iloc[-1]
        forecast = forecast_from_dataset(df)
        dataset_result = solution(forecast, last_row=last_row)

        # ---------- FINAL RESPONSE ----------
        return JSONResponse({
            "image_analysis": image_result,
            "dataset_analysis": dataset_result
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import models, layers, applications
from tensorflow.keras.preprocessing import image

app = FastAPI()

# Allow React frontend to call FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080","http://localhost:8000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Model Architecture
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
model = models.Model(inputs, outputs)

# Load pretrained weights
model.load_weights("dominator.h5")

# -----------------------------
# Label and solution dictionaries
# -----------------------------
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
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    # Simple HTML form for testing
    content = """
    <html>
        <head><title>Upload Image</title></head>
        <body>
            <h2>Upload a Leaf Image for Prediction</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid file format. Upload JPG or PNG.")

    try:
        # Load and preprocess image
        img = Image.open(file.file).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict once
        pred = model.predict(img_array)
        top_indices = pred.argsort(axis=1)[0, -3:][::-1]  # Top 3 predictions

        top_labels = [label_dict[i] for i in top_indices]
        top_solutions = [solution_dict[i] for i in top_indices]

        # Construct response
        if top_labels[0] == "Healthy":
            response = {
                "Status": "Healthy",
                "message": f"The Sugarcane leaf is {top_labels[0]}",
                "solutions": [top_solutions[0]],  # <-- list now
                "confidence": f"{np.max(pred)*100:.2f}"
            }
        else:
            response = {
                "Status": "Infected",
                "message": f"The Sugarcane leaf is infected by {top_labels[0]} and {top_labels[1]}",
                "solutions": top_solutions[:2],
                "confidence": f"{np.max(pred)*100:.2f}"
            }

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

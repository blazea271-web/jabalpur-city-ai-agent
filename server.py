import os
import io
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok

# 1. Initialize FastAPI app & add CORS for FlutterFlow
app = FastAPI(title="Jabalpur AI City Guide API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load your trained CNN model
print("🧠 Loading CNN model from 'jabalpur_cnn.keras'...")
model = tf.keras.models.load_model("jabalpur_cnn.keras")
class_names = ["dhuandhar", "gwarighat", "madan_mahal" ,"siddhbaba"]

# 3. Google Maps coordinates and details
PLACE_DETAILS = {
    "dhuandhar": {
        "name": "Dhuandhar Falls",
        "category": "Waterfall / Nature",
        "lat": 23.1299,
        "lng": 79.8058,
        "description": "Massive waterfall on the Narmada River creating a smoke-like mist. Famous for cable car ropeway rides.",
        "timings": "6:00 AM - 8:00 PM"
    },
    "madan_mahal": {
        "name": "Madan Mahal Fort",
        "category": "Historical Monument",
        "lat": 23.1517,
        "lng": 79.9077,
        "description": "11th-century Gond watchtower fort on a rocky hill with 360-degree views of Jabalpur city.",
        "timings": "7:00 AM - 6:30 PM"
    },
    "gwarighat": {
    "name": "Gwarighat",
    "category": "Riverfront / Religious Place",
    "lat": 23.10806,
    "lng": 79.92833,
    "description": "Gwarighat is a famous Narmada River ghat in Jabalpur, known for the Narmada Aarti, riverside temples, ghats and boating.",
    "timings": "Open throughout the day",
    },
    "siddhbaba": {
    "name": "Siddh Baba Temple",
    "category": "Religious Place / Hindu Temple",
    "lat": 23.1676,
    "lng": 79.9185,
    "description": "Siddh Baba Temple is a peaceful Hindu temple in the Lalmati area of Jabalpur, known for its spiritual atmosphere and panoramic views of the surrounding city.",
    "timings": "8:00 AM - 11:00 PM"

},
}

# 4. The /identify endpoint
@app.post("/identify")
async def identify_landmark(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((128, 128))
    
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions[0]) * 100)

    place_info = PLACE_DETAILS.get(predicted_class, {})

    return {
        "status": "success",
        "predicted_id": predicted_class,
        "name": place_info.get("name"),
        "confidence": f"{confidence:.1f}%",
        "category": place_info.get("category"),
        "lat": place_info.get("lat"),
        "lng": place_info.get("lng"),
        "description": place_info.get("description"),
        "timings": place_info.get("timings")
    }

# 5. One-Click Startup: Opens tunnel + starts server
if __name__ == "__main__":
    PORT = 8000

    ngrok.set_auth_token("37h5u2DqJYm1nU4zdsxg1Rt0vVE_5gEgyptdwpqMXMb4j7boN")
    
    # Automatically open ngrok tunnel
    print("🌐 Connecting public tunnel...")
    tunnel = ngrok.connect(PORT)
    public_url = tunnel.public_url

    print("\n" + "="*60)
    print("🚀 YOUR LIVE FLUTTERFLOW API URL IS:")
    print(f"👉 {public_url}/identify")
    print("="*60 + "\n")

    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=PORT)
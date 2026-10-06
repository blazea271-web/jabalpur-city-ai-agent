# 🧠 Jabalpur City AI Agent

A **deep-learning-powered landmark identification and city exploration system** for Jabalpur.

The core of the project is a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** that identifies Jabalpur landmarks from images. The trained model is served through a **FastAPI backend** and integrated into a **FlutterFlow application**.

The application also includes a **Gemini-powered AI chatbot** to provide conversational assistance to users exploring the city.

---

## 🧠 Deep Learning & Computer Vision

The primary component of this project is a **CNN-based image classification system** developed using TensorFlow and Keras.

The model is trained to recognize different landmarks in Jabalpur from images.

### 🏛️ Currently Supported Landmarks

- Dhuandhar Falls
- Gwarighat
- Madan Mahal Fort
- Siddh Baba Temple

### 🧠 CNN Architecture

The model uses a sequential CNN architecture consisting of:

```text
Input Image
     ↓
Rescaling
     ↓
Conv2D (32 filters)
     ↓
MaxPooling2D
     ↓
Conv2D (64 filters)
     ↓
MaxPooling2D
     ↓
Conv2D (64 filters)
     ↓
MaxPooling2D
     ↓
Flatten
     ↓
Dense (64 neurons)
     ↓
Softmax Output
```

### ⚙️ Model Configuration

| Parameter | Value |
|---|---|
| Image Size | 128 × 128 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Output Activation | Softmax |
| Framework | TensorFlow / Keras |

The number of output neurons is determined dynamically from the number of landmark classes in the training dataset.

### 📂 Training Dataset

The dataset is organized into class-specific directories:

```text
dataset/
├── dhuandhar/
├── gwarighat/
├── madan_mahal/
└── siddhbaba/
```

Each directory contains images belonging to its respective landmark class.

### 🔍 Prediction Process

```text
Landmark Image
      ↓
Image Preprocessing
      ↓
Resize to 128 × 128
      ↓
Pixel Rescaling
      ↓
Trained CNN
      ↓
Class Probabilities
      ↓
Highest Probability Class
      ↓
Predicted Landmark
```

The model uses `softmax` to generate probabilities for each landmark class. The class with the highest probability is selected as the predicted landmark.

---

## ⚙️ FastAPI Backend

The trained CNN model is integrated into a **FastAPI backend**.

FastAPI acts as the bridge between the machine learning model and the application frontend.

### 🔌 Main Endpoint

```text
POST /identify
```

The endpoint accepts an uploaded image and sends it through the trained CNN model.

### 🔄 Backend Workflow

```text
Image Upload
     ↓
FastAPI
     ↓
Image Reading
     ↓
RGB Conversion
     ↓
Resize to 128 × 128
     ↓
CNN Prediction
     ↓
Prediction Confidence
     ↓
Landmark Information
     ↓
JSON Response
```

### 📤 Example Response

```json
{
  "status": "success",
  "predicted_id": "dhuandhar",
  "name": "Dhuandhar Falls",
  "confidence": "95.4%",
  "category": "Waterfall / Nature",
  "lat": 23.1299,
  "lng": 79.8058,
  "description": "Massive waterfall on the Narmada River.",
  "timings": "6:00 AM - 8:00 PM"
}
```

The backend returns both the CNN prediction and additional information associated with the predicted landmark.

---

## 🎨 FlutterFlow Frontend

The application interface is developed using **FlutterFlow**.

FlutterFlow provides the user-facing interface through which users interact with the AI system.

### 📱 Main Features

- Landmark image upload
- AI landmark identification
- Landmark information display
- Maps
- AI chatbot interface
- User profile
- Saved places
- History
- Navigation between application sections

### 🖼️ Landmark Identification Flow

```text
User
 ↓
FlutterFlow
 ↓
Upload / Capture Image
 ↓
FastAPI API
 ↓
CNN Model
 ↓
Prediction
 ↓
FlutterFlow Result Screen
```

The web and mobile interfaces provide users with a simple way to upload a landmark image and view the model's prediction.

---

## 🤖 Gemini AI Chatbot

The application also contains an AI-powered chatbot integrated using the **Gemini API**.

The chatbot is implemented through the FlutterFlow application and provides conversational assistance for users exploring the city.

### 💬 Chatbot Flow

```text
User Question
      ↓
FlutterFlow Chat Interface
      ↓
Gemini API
      ↓
AI Generated Response
      ↓
FlutterFlow
      ↓
User
```

The Gemini API is used separately from the CNN landmark classification system.

The CNN handles **visual landmark recognition**, while Gemini handles **conversational AI interactions**.

> API keys and other sensitive credentials are not included in this repository.

---

## 🏗️ Overall System Architecture

```text
                         JABALPUR CITY AI AGENT
                                  │
                                  │
                         ┌────────▼────────┐
                         │   FlutterFlow   │
                         │    Frontend     │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             Landmark Image                AI Chatbot
                    │                           │
                    ▼                           ▼
             FastAPI Backend                Gemini API
                    │
                    ▼
             TensorFlow / Keras
                    │
                    ▼
                  CNN
                    │
                    ▼
          Landmark Classification
                    │
                    ▼
          Landmark Information
                    │
                    ▼
              FlutterFlow
```

---

## 🔬 Machine Learning Pipeline

The complete computer vision pipeline is:

```text
Dataset
   ↓
Image Loading
   ↓
Image Resizing
   ↓
CNN Training
   ↓
Model Evaluation
   ↓
Saved Keras Model
   ↓
FastAPI Integration
   ↓
Image Prediction
   ↓
Landmark Classification
```

The trained model is saved as:

```text
jabalpur_cnn.keras
```

---

## 🛠️ Technologies Used

### 🧠 Deep Learning & AI

- Python
- TensorFlow
- Keras
- Convolutional Neural Networks (CNN)
- Computer Vision
- Gemini API

### ⚙️ Backend

- FastAPI
- Uvicorn
- Pillow
- NumPy

### 🎨 Frontend & Application

- FlutterFlow
- Firebase
- REST API integration

### 💻 Development & Version Control

- VS Code
- Git
- GitHub

---

## 📁 Project Structure

```text
jabalpur-city-ai-agent/
│
├── dataset/
│   ├── dhuandhar/
│   ├── gwarighat/
│   ├── madan_mahal/
│   └── siddhbaba/
│
├── jabalpur_cnn.keras
├── server.py
├── train.py
├── test_data.py
├── .gitignore
└── README.md
```

---

## ▶️ Running the Deep Learning Backend

### 1. Clone the Repository

```bash
git clone https://github.com/blazea271-web/jabalpur-city-ai-agent.git
```

### 2. Enter the Project Directory

```bash
cd jabalpur-city-ai-agent
```

### 3. Install Dependencies

```bash
pip install tensorflow fastapi uvicorn pillow numpy python-multipart
```

### 4. Start the FastAPI Server

```bash
uvicorn server:app --reload
```

### 5. Open FastAPI Documentation

```text
http://127.0.0.1:8000/docs
```

The interactive Swagger documentation can be used to test the `/identify` endpoint.

---

## 📸 Landmark Identification Example

```text
                    INPUT
                      │
                      ▼
               ┌─────────────┐
               │ Landmark    │
               │ Image       │
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │ CNN Model   │
               └──────┬──────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Predicted Class │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Landmark Data   │
             └────────┬────────┘
                      │
                      ▼
                Final Result
```

---

## 🎯 Project Objective

The objective of the **Jabalpur City AI Agent** is to combine **deep learning, computer vision, backend APIs, generative AI, and application development** into a single city exploration platform.

The primary focus is the CNN-based landmark identification system, while the FastAPI backend, FlutterFlow application, maps, Firebase, and Gemini chatbot provide the supporting infrastructure for the complete AI travel experience.

---

## 🔮 Future Improvements

Potential improvements include:

- Increase the number of landmark classes
- Expand the training dataset
- Use transfer learning with a pretrained CNN architecture
- Improve model accuracy and generalization
- Add image augmentation
- Add more city-specific information
- Improve landmark confidence handling
- Integrate additional travel recommendations
- Add route and navigation features
- Deploy the machine learning API to the cloud

---

## 👨‍💻 Developer

**Anish John**

B.Tech — Artificial Intelligence & Machine Learning

---

## 🔒 Security

API keys, authentication credentials, environment variables, and other sensitive information are intentionally excluded from this repository.

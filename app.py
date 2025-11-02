from flask import Flask, request, jsonify, send_file
import pickle
import pandas as pd
import os

# Load model
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl not found! Make sure it exists in project root.")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# Home route
@app.route("/", methods=["GET"])
def home():
    return "ML API is running! Visit /frontend for browser interface or use POST /predict."

# Front-end route
@app.route("/frontend", methods=["GET"])
def frontend():
    return send_file("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

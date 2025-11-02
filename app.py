from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load your trained ML model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception("model.pkl not found! Make sure it's in the same folder as app.py")

app = Flask(__name__)

# Root route to check if API is running
@app.route("/", methods=["GET"])
def home():
    return "ML API is running! Use POST /predict for predictions."

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

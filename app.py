from flask import Flask, request, jsonify
import pandas as pd
import statsmodels.api as sm
import pickle
import os

app = Flask(__name__)

# Load trained models
models = {}
for file in os.listdir():
    if file.startswith("arima_") and file.endswith(".pkl"):
        material = file.split("_")[1].split(".")[0]
        with open(file, "rb") as f:
            models[material] = pickle.load(f)

@app.route('/predict', methods=['GET'])
def predict():
    material = request.args.get("material_name")
    steps = int(request.args.get("steps", 1))  # Number of months to predict

    if material not in models:
        return jsonify({"error": "Material not found"}), 404

    model = models[material]
    forecast = model.forecast(steps=steps)

    return jsonify({
        "material": material,
        "forecast": list(forecast)
    })

if __name__ == "__main__":
    app.run(debug=True)

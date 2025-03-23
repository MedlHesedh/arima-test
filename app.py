from flask import Flask, request, jsonify
import pandas as pd
import statsmodels.api as sm
import pickle
import os
import base64
from train import train_models

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Load trained models
models = {}
material_mapping = {}  # Map encoded names back to original names

for file in os.listdir():
    if file.startswith("arima_") and file.endswith(".pkl"):
        encoded_material = file.split("_")[1].split(".")[0]
        try:
            # Decode the base64 encoded material name
            material = base64.b64decode(encoded_material.encode()).decode()
            with open(file, "rb") as f:
                models[material] = pickle.load(f)
            material_mapping[material] = encoded_material
        except:
            # Skip files that can't be properly decoded
            continue

@app.route('/predict', methods=['GET'])
def predict():
    material = request.args.get("material_name")
    print(f"Predicting for material: {material}")
    steps = int(request.args.get("steps", 1))  # Number of months to predict

    if material not in models:
        return jsonify({"error": "Material not found"}), 404

    model = models[material]
    forecast = model.forecast(steps=steps)

    return jsonify({
        "material": material,
        "forecast": list(forecast)
    })

@app.route('/train', methods=['GET'])
def train():
    connection_string = request.args.get("connection_string", "")
    
    # Call the training function
    success, results = train_models(connection_string)
    
    # Reload models after training
    global models, material_mapping
    models = {}
    material_mapping = {}
    
    for file in os.listdir():
        if file.startswith("arima_") and file.endswith(".pkl"):
            encoded_material = file.split("_")[1].split(".")[0]
            try:
                # Decode the base64 encoded material name
                material = base64.b64decode(encoded_material.encode()).decode()
                with open(file, "rb") as f:
                    models[material] = pickle.load(f)
                material_mapping[material] = encoded_material
            except:
                # Skip files that can't be properly decoded
                continue
    
    if success:
        return jsonify({
            "status": "success",
            "message": "Models trained successfully",
            "details": results
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Error in training process",
            "details": results
        }), 500

if __name__ == "__main__":
    app.run(debug=True)

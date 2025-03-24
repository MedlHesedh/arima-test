from flask import Flask, request, jsonify
import pandas as pd
import statsmodels.api as sm
import pickle
import os
import base64
from train import train_models
from supabase_config import supabase

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# No longer need to load models at startup since we'll load from Supabase on demand
models = {}
material_mapping = {}

@app.route('/predict', methods=['GET'])
def predict():
    item_type = request.args.get("type")  # 'material' or 'labor'
    item_name = request.args.get("name")
    print(f"Predicting for {item_type}: {item_name}")
    steps = int(request.args.get("steps", 1))

    full_name = f"{item_type}_{item_name}"
    model_filename = f"arima_{full_name}.pkl"

    # Download the model from Supabase
    try:
        res = supabase.storage.from_('arima-models').download(model_filename)
        if not res or 'error' in res:
            return jsonify({"error": f"{item_type} not found"}), 404

        # Load the ARIMA model
        model = pickle.loads(res)
        
        # Generate forecast
        forecast = model.forecast(steps=steps)

        return jsonify({
            "type": item_type,
            "name": item_name,
            "forecast": forecast.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['GET'])
def train():
    connection_string =  f'postgresql://postgres.viculrdtittnlgikngxg:9ouZiUP4JK6F45ST@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres'
    
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

@app.route('/train', methods=['POST'])
def train_single():
    material_name = request.json.get('material')
    data = load_data_for_material(material_name)  # Replace with your data loading logic

    # Train ARIMA model
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(data, order=(1, 1, 1)).fit()

    # Save model to a pickle file
    model_filename = f"arima_{material_name}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    # Upload to Supabase Storage
    with open(model_filename, "rb") as f:
        res = supabase.storage.from_('arima-models').upload(model_filename, f)

    if 'error' in res:
        return jsonify({"error": res['error']}), 500

    return jsonify({"message": "Model trained and uploaded to Supabase"})

if __name__ == "__main__":
    app.run(debug=True)

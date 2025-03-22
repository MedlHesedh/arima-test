import pandas as pd
import statsmodels.api as sm
import pickle

# Load dataset
df = pd.read_csv("materials_prices.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# Dictionary to store models
trained_models = {}

# Train an ARIMA model for each material
for material in df["material_name"].unique():
    data = df[df["material_name"] == material]["price"]
    
    # Fit ARIMA model (you can tune p, d, q)
    model = sm.tsa.ARIMA(data, order=(2,1,2))  # ARIMA(p,d,q)
    model_fit = model.fit()
    
    # Save the model
    with open(f"arima_{material}.pkl", "wb") as f:
        pickle.dump(model_fit, f)
    
    trained_models[material] = model_fit

print("Models trained and saved.")

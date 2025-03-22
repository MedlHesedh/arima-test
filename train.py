import pandas as pd
import statsmodels.api as sm
import pickle
from sqlalchemy import create_engine

# Database connection parameters
db_params = {
    'host': 'localhost',
    'database': 'materials_db',
    'user': 'postgres',
    'password': 'your_password',
    'port': '5432'
}

# Load dataset from PostgreSQL

engine = create_engine(f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}')
df = pd.read_sql('SELECT date, material_name, price FROM materials_prices', engine, parse_dates=["date"])
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

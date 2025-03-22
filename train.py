import pandas as pd
import statsmodels.api as sm
import pickle
from sqlalchemy import create_engine
import base64

# Database connection parameters
db_params = {
    'host': 'localhost',
    'database': 'materials_db',
    'user': 'postgres',
    'password': 'your_password',
    'port': '5432'
}

# Load dataset from PostgreSQL
engine = create_engine(
    f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}',
    connect_args={'sslmode': 'verify-full', 'sslcert': '/path/to/client-cert.pem'}
)
df = pd.read_sql('SELECT date, cost, material FROM material_history', engine, parse_dates=["date"])
df.set_index("date", inplace=True)

# Dictionary to store models
trained_models = {}

# Train an ARIMA model for each material
for material in df["material"].unique():
    data = df[df["material"] == material]["cost"]
    
    # Fit ARIMA model (you can tune p, d, q)
    model = sm.tsa.ARIMA(data, order=(2,1,2))  # ARIMA(p,d,q)
    model_fit = model.fit()
    
    # Encode material name for safe file naming
    encoded_material = base64.b64encode(material.encode()).decode()
    
    # Save the model
    with open(f"arima_{encoded_material}.pkl", "wb") as f:
        pickle.dump(model_fit, f)
    
    trained_models[material] = model_fit

print("Models trained and saved.")

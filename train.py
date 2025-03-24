import pandas as pd
import statsmodels.api as sm
import pickle
from sqlalchemy import create_engine
import base64

def train_models(connection_string=""):
    # Dictionary to store models
    trained_models = {}
    results = {"success": [], "errors": []}
    
    try:
        # Load datasets from PostgreSQL
        engine = create_engine(connection_string, connect_args={"sslmode": "disable"})
        
        # Load material data
        material_df = pd.read_sql('SELECT date, cost, material FROM material_history', engine, parse_dates=["date"])
        material_df.set_index("date", inplace=True)
        
        # Load labor data
        labor_df = pd.read_sql('SELECT date, cost, labor FROM labor_history', engine, parse_dates=["date"])
        labor_df.set_index("date", inplace=True)
        
        # Train models for materials
        for material in material_df["material"].unique():
            try:
                data = material_df[material_df["material"] == material]["cost"]
                model = sm.tsa.ARIMA(data, order=(2, 1, 2))
                model_fit = model.fit()
                
                encoded_name = base64.b64encode(f"material_{material}".encode()).decode()
                
                with open(f"arima_{encoded_name}.pkl", "wb") as f:
                    pickle.dump(model_fit, f)
                
                trained_models[f"material_{material}"] = model_fit
                results["success"].append(f"material_{material}")
                print(f"Successfully trained model for material: {material}")
            except Exception as e:
                error_msg = f"Error training model for material {material}: {str(e)}"
                results["errors"].append({"item": f"material_{material}", "error": str(e)})
                print(error_msg)
                continue
        
        # Train models for labor
        for labor in labor_df["labor"].unique():
            try:
                data = labor_df[labor_df["labor"] == labor]["cost"]
                model = sm.tsa.ARIMA(data, order=(2, 1, 2))
                model_fit = model.fit()
                
                encoded_name = base64.b64encode(f"labor_{labor}".encode()).decode()
                
                with open(f"arima_{encoded_name}.pkl", "wb") as f:
                    pickle.dump(model_fit, f)
                
                trained_models[f"labor_{labor}"] = model_fit
                results["success"].append(f"labor_{labor}")
                print(f"Successfully trained model for labor: {labor}")
            except Exception as e:
                error_msg = f"Error training model for labor {labor}: {str(e)}"
                results["errors"].append({"item": f"labor_{labor}", "error": str(e)})
                print(error_msg)
                continue
        
        print("Models trained and saved.")
        return True, results
    except Exception as e:
        print(f"Error in training process: {str(e)}")
        return False, {"error": str(e)}

# If run directly as a script
if __name__ == "__main__":
    success, results = train_models()
    print("Training completed with status:", "Success" if success else "Failed")

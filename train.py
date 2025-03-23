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
        # Load dataset from PostgreSQL
        engine = create_engine(connection_string, connect_args={"sslmode": "disable"})
        df = pd.read_sql("SELECT * FROM sales_data", engine)
        df.set_index("date", inplace=True)
        
        # Train an ARIMA model for each material
        for material in df["material"].unique():
            try:
                data = df[df["material"] == material]["cost"]
                
                # Fit ARIMA model (you can tune p, d, q)
                model = sm.tsa.ARIMA(data, order=(2, 1, 2))  # ARIMA(p,d,q)
                model_fit = model.fit()
                
                # Encode material name for safe file naming
                encoded_material = base64.b64encode(material.encode()).decode()
                
                # Save the model
                with open(f"arima_{encoded_material}.pkl", "wb") as f:
                    pickle.dump(model_fit, f)
                
                trained_models[material] = model_fit
                results["success"].append(material)
                print(f"Successfully trained model for {material}")
            except Exception as e:
                error_msg = f"Error training model for {material}: {str(e)}"
                results["errors"].append({"material": material, "error": str(e)})
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

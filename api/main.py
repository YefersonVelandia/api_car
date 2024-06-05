from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargar el modelo entrenado y las columnas
model = joblib.load('https://grupoalianzacolombia.com/model/car_modelo.pkl')
model_columns = joblib.load('https://grupoalianzacolombia.com/model/modelo_car_columnas.pkl')

# Definir la aplicación FastAPI
app = FastAPI()

# Definir el modelo de datos para la entrada
class Vehicle(BaseModel):
    year: int
    make: str
    model: str
    body: str
    color: str
    interior: str
    transmission: str
    state: str
    condition: float
    odometer: float

# Definir el endpoint para predecir el precio
@app.post("/predict")
def predict_price(vehicle: Vehicle):
    # Convertir los datos de entrada en un DataFrame
    data = {
        'year': [vehicle.year],
        'make': [vehicle.make],
        'model': [vehicle.model],
        'body': [vehicle.body],
        'color': [vehicle.color],
        'interior': [vehicle.interior],
        'transmission': [vehicle.transmission],
        'state': [vehicle.state],
        'condition': [vehicle.condition],
        'odometer': [vehicle.odometer],
    }
    
    df = pd.DataFrame(data)
    df = pd.get_dummies(df)
    
    # Asegurarse de que el DataFrame tenga las mismas columnas que las que se usaron para entrenar el modelo
    missing_cols = set(model_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[model_columns]
    
    # Hacer la predicción
    prediction = model.predict(df)
    
    return {"predicted_price": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

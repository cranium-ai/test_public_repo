from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained model
model_file = "trained_model.joblib"
model = joblib.load(model_file)

# Define the API app
app = FastAPI()

# Define the input schema using Pydantic BaseModel
class PurchaseData(BaseModel):
    age: int
    location: str
    page_name: str
    time_on_page: float

# Define the prediction endpoint
@app.post("/predict")
def predict_purchase(data: PurchaseData):
    # Convert the input to a dictionary
    data_dict = data.dict()

    # Prepare the input for prediction
    input_data = [[data_dict['age'], data_dict['location'], data_dict['page_name'], data_dict['time_on_page']]]

    # Make the prediction
    prediction = model.predict(input_data)

    # Prepare the response
    response = {
        'prediction': bool(prediction[0])
    }

    return response

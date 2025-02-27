from fastapi import FastAPI, HTTPException
import torch
import joblib
import mlflow.pytorch
from pydantic import BaseModel

# Load your GNN model (e.g., the student model for this example)
MODEL_URI = "models:/student_model/Production"  # or use torch.load if saved locally
model = mlflow.pytorch.load_model(MODEL_URI)
model.eval()

# Load the pre-fitted KMeans model (used to compute the cluster)
kmeans = joblib.load('kmeans_model.pkl')

app = FastAPI()

# Define the expected request body
class PredictionRequest(BaseModel):
    features: list  # expects a list of 4 numbers [tick, cape, cattle, bio5]
    use_teacher_model: bool = False  # optional flag if you want to choose between models

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Ensure the user provided 4 features
        if len(request.features) != 4:
            raise HTTPException(status_code=400, detail="Exactly 4 features are required: tick, cape, cattle, bio5.")

        # Compute the cluster using the pre-fitted KMeans model
        cluster_label = int(kmeans.predict([request.features])[0])
        
        # Combine the raw features with the computed cluster to form a 5-element input vector
        input_features = request.features + [cluster_label]
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
        
        # Create a dummy edge index for a single node graph.
        # Here, we create a self-loop on the single node: edge_index = [[0],[0]]
        dummy_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor, dummy_edge_index)  # Ensure edge_index is passed here
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.max(torch.exp(output), dim=1).values.item()  # Assuming output is in log-probabilities
        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "model_used": "Teacher" if request.use_teacher_model else "Student"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
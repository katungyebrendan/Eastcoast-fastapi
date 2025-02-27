from fastapi import FastAPI, HTTPException
import torch
import joblib
from pydantic import BaseModel
import os

# Create FastAPI app
app = FastAPI()

# Load models from local paths
try:
    model_path = os.path.join("models", "student_model.pth")
    model = torch.load(model_path)
    model.eval()
    
    kmeans_path = os.path.join("models", "kmeans_model.pkl")
    kmeans = joblib.load(kmeans_path)
except Exception as e:
    print(f"Error loading models: {str(e)}")
    # Continue anyway so the app starts - we'll handle model errors later

# Define request model
class PredictionRequest(BaseModel):
    features: list  # expects a list of 4 numbers [tick, cape, cattle, bio5]
    use_teacher_model: bool = False

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
        
        # Create a dummy edge index for a single node graph
        dummy_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor, dummy_edge_index)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.max(torch.exp(output), dim=1).values.item()
        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "model_used": "Teacher" if request.use_teacher_model else "Student"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a simple health check endpoint
@app.get("/")
def read_root():
    return {"status": "API is running"}

# If this file is run directly, start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

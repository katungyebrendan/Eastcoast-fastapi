from fastapi import FastAPI, HTTPException
import torch
import joblib
from pydantic import BaseModel
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Load models from local paths
model, kmeans = None, None  # Initialize as None for error handling

try:
    model_path = os.path.join("models", "student_model.pth")
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Load on CPU for compatibility
    model.eval()
    logger.info("Student model loaded successfully.")

    kmeans_path = os.path.join("models", "kmeans_model.pkl")
    kmeans = joblib.load(kmeans_path)
    logger.info("KMeans model loaded successfully.")

except Exception as e:
    logger.error(f"Error loading models: {str(e)}")

# Define request model
class PredictionRequest(BaseModel):
    features: list  # Expects a list of 4 numbers [tick, cape, cattle, bio5]
    use_teacher_model: bool = False

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        if model is None or kmeans is None:
            raise HTTPException(status_code=500, detail="Models are not loaded properly.")

        # Ensure the user provided exactly 4 features
        if len(request.features) != 4:
            raise HTTPException(status_code=400, detail="Exactly 4 features are required: tick, cape, cattle, bio5.")
        
        # Compute the cluster using the pre-fitted KMeans model
        cluster_label = int(kmeans.predict([request.features])[0])
        
        # Combine the raw features with the computed cluster to form a 5-element input vector
        input_features = request.features + [cluster_label]
        input_tensor = torch.tensor([input_features], dtype=torch.float32)  # Ensure proper shape
        
        # Create a dummy edge index for a single-node graph
        dummy_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor, dummy_edge_index)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1).values.item()
        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "model_used": "Teacher" if request.use_teacher_model else "Student"
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "API is running"}

# Start the server only when running directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))  # Ensure it matches Render's assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)

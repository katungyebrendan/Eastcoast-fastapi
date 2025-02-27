import torch
import os
import logging
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Define the model architecture (must match original structure)
class StudentModel(torch.nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc1 = torch.nn.Linear(5, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.fc3 = torch.nn.Linear(8, 2)  # Assuming binary classification

    def forward(self, x, edge_index):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load models
model, kmeans = None, None

try:
    # Load KMeans model
    kmeans_path = os.path.join("models", "kmeans_model.pkl")
    kmeans = joblib.load(kmeans_path)
    logger.info("KMeans model loaded successfully.")

    # Load PyTorch model correctly
    model_path = os.path.join("models", "student_model.pth")
    model = StudentModel()  # Initialize model first
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Put model in evaluation mode
    logger.info("Student model loaded successfully.")

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

        # Ensure exactly 4 features
        if len(request.features) != 4:
            raise HTTPException(status_code=400, detail="Exactly 4 features required: tick, cape, cattle, bio5.")
        
        # Predict cluster using KMeans
        cluster_label = int(kmeans.predict([request.features])[0])
        
        # Create input tensor
        input_features = request.features + [cluster_label]
        input_tensor = torch.tensor([input_features], dtype=torch.float32)
        
        # Dummy edge index for GCN compatibility
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))  
    uvicorn.run(app, host="0.0.0.0", port=port)

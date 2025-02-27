import numpy as np  # Fix the missing 'i'
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class StudentModel(torch.nn.Module):
    """
    A Graph Neural Network model with two graph convolutional layers.
    This architecture matches the structure found in the student_model.pth file.
    """
    def __init__(self, input_features, hidden_channels, output_features):
        """
        Initialize the model with configurable layer sizes.
        
        Args:
            input_features: Number of input features per node
            hidden_channels: Size of hidden representation after first conv layer
            output_features: Number of output features per node
        """
        super(StudentModel, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_features)
    
    def forward(self, x, edge_index):
        """
        Forward pass through the network.
        
        Args:
            x: Node feature matrix with shape [num_nodes, input_features]
            edge_index: Graph connectivity in COO format with shape [2, num_edges]
            
        Returns:
            Output node features
        """
        # First graph convolution layer followed by ReLU activation
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        
        # Second graph convolution layer
        x = self.conv2(x, edge_index)
        
        return x

def load_model(path="student_model.pth", input_features=32, hidden_channels=64, output_features=1):
    """
    Load the saved model weights into the model architecture.
    
    Args:
        path: Path to the saved model file
        input_features: Number of input features per node
        hidden_channels: Size of hidden representation
        output_features: Number of output features per node
        
    Returns:
        Loaded model with pre-trained weights
    """
    # Create model with the same architecture
    model = StudentModel(input_features, hidden_channels, output_features)
    
    # Load the state dictionary from the .pth file
    state_dict = torch.load(path)
    
    # Load the weights into the model
    model.load_state_dict(state_dict)
    
    return model

def example_usage():
    """
    Example showing how to use the model for inference.
    """
    # Create a random graph for demonstration
    num_nodes = 10
    input_features = 32
    
    # Random node features
    x = torch.randn(num_nodes, input_features)
    
    # Random edges (just for example)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                              [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)
    
    # Load the model
    model = load_model(input_features=input_features)
    
    # Set to evaluation mode
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        output = model(x, edge_index)
    
    print(f"Output shape: {output.shape}")
    print("Output predictions:", output)

if __name__ == "__main__":
    example_usage()

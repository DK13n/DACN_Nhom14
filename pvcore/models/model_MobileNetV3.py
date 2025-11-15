import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models
import torch
import os

from pvcore.shared.config import device , num_frames

class CNNTemporalAvgPooling(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNTemporalAvgPooling, self).__init__()
        self.cnn = models.mobilenet_v3_large(pretrained=True)
        self.cnn.classifier = nn.Identity()  
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()  
        out_decision_t = []
        cnn_features = []
        
        for t in range(seq_len):
            feature = self.cnn(x[:, t, :, :, :])  
            cnn_features.append(feature)
         
        cnn_features = torch.stack(cnn_features, dim=1)  
        temporal_avg_features = cnn_features.mean(dim=1)  
        out = self.fc(temporal_avg_features)  
        return out

    def extract_intermediate_features(self, x):
        """Extract features before and after LSTM."""
        batch_size, seq_len, C, H, W = x.size()
        cnn_features = []
        for t in range(seq_len):
            feature = self.cnn(x[:, t, :, :, :])  
            cnn_features.append(feature)
        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: [batch_size, seq_len, cnn_output_dim]
        temporal_avg_features = cnn_features.mean(dim=1)  # Temporal average pooling: Shape: [batch_size, 960]
        return cnn_features, temporal_avg_features
    
def get_model( weights_path=None):

    model = CNNTemporalAvgPooling(num_classes=2).to(device)

    if weights_path is None:
        base_dir = os.path.dirname(__file__)
        weights_path = os.path.join(base_dir, "weights", "MobileNetV3.pth")

    print(f"Loading model from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")

    return model

if __name__ == "__main__":
    model = get_model()
    print('ok')
    #python3 -m pvcore.models.model_MobileNetV3
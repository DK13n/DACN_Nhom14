import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models
import torch
import os

from pvcore.shared.config import device , num_frames

class ResNetBranch(nn.Module):
    def __init__(self, input_dim=512, output_dim=256, pretrained=True):
        super(ResNetBranch, self).__init__()
        self.resnet = timm.create_model('resnet18', pretrained=pretrained, num_classes=0)
        feature_dim = self.resnet.num_features
        
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        batch_size, num_frames = x.shape[0], x.shape[1]

        x = x.view(batch_size * num_frames, *x.shape[2:])
        features = self.resnet(x)
        features = self.projector(features)

        features = features.view(batch_size, num_frames, -1)
        temporal_features = torch.mean(features, dim=1)  
        
        return temporal_features

class ViTBranch(nn.Module):
    def __init__(self, input_dim=512, output_dim=256, num_frames=16, pretrained=True):
        super(ViTBranch, self).__init__()
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=0)
        vit_dim = self.vit.num_features
        self.input_projection = nn.Linear(input_dim, vit_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_dim))
        
        self.output_projection = nn.Linear(vit_dim, output_dim)
        
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        x = self.input_projection(x)  

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  
        
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device)
            extended_mask = torch.cat((cls_mask, mask), dim=1)
            padding_mask = ~extended_mask.bool()
        else:
            padding_mask = None

        x = self.temporal_encoder(x, src_key_padding_mask=padding_mask)
        cls_output = x[:, 0]  # B x vit_dim
        output = self.output_projection(cls_output)  # B x output_dim
        
        return output
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CDCN(nn.Module):
    """CDCN feature extractor với residual connections"""
    def __init__(self, in_channels=3, feature_dim=512):
        super(CDCN, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        batch_size, num_frames = x.shape[0], x.shape[1]

        x = x.view(batch_size * num_frames, *x.shape[2:])
        
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = x.view(batch_size, num_frames, -1)
        
        return x




class CDRes_ViT(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512, branch_dim=256, num_frames=16, alpha=0.5):
        super(CDRes_ViT, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))  # Learnable alpha
        
        self.cdcn = CDCN(feature_dim=feature_dim)
        self.resnet_branch = ResNetBranch(input_dim=feature_dim, output_dim=branch_dim)
        self.vit_branch = ViTBranch(input_dim=feature_dim, output_dim=branch_dim, num_frames=num_frames)
        
        self.classifier = nn.Sequential(
            nn.Linear(branch_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, mask=None):

        cdcn_features = self.cdcn(x)  
        resnet_features = self.resnet_branch(x)  
        vit_features = self.vit_branch(cdcn_features, mask)  
        combined_features = self.alpha * resnet_features + (1 - self.alpha) * vit_features
        logits = self.classifier(combined_features)
        return logits
    
     

def get_model( weights_path=None):

    model = CDRes_ViT(
        num_classes=2,
        feature_dim=900,
        branch_dim=256,
        num_frames= num_frames,  # lấy từ config
        alpha=0.5
    ).to(device)

    if weights_path is None:
        base_dir = os.path.dirname(__file__)
        weights_path = os.path.join(base_dir, "weights", "Hybrid-CDCN-ResViT.pth")

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
    #python3 -m pvcore.models.model_VisionTriX
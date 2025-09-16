import torch
from torch import nn
import torch.nn.functional as F
import math
from src.Model_arch.CDCN import CDCN
from src.Model_arch.ResNet import ResNet
from src.Model_arch.ViT import TemporalViT


class CDRes_ViT(nn.Module):
    """
    - CDCN: trích xuất fmap
    - ViT:   B,T,C,H,W → CLS (B,D)
    - ResNet: ảnh hoặc video → embedding (B,D) (video mặc định pool theo T='mean')
    - Fusion: weighted sum (alpha ∈ (0,1), learnable)
    """
    def __init__(self,
                 device,
                 cdcn_cfg=dict(in_ch=3, stem_out=32, c1=64, c2=128, theta=0.7, use_maxpool=True),
                 d_model=256,
                 vit_cfg=dict(depth=4, num_heads=8, mlp_ratio=4.0, dropout=0.1, max_len=256),
                 resnet_cfg=dict(layers=(2,2,2,2)),
                 ):
        super().__init__()
        self.device = device
        # CDCN backbone
        self.cdcn = CDCN(**cdcn_cfg)
        C_cdc = cdcn_cfg.get("c2", 128)

        # ViT branch (yêu cầu đầu vào 5D B,T,C,H,W)
        self.vit  = TemporalViT(in_channels=C_cdc, embed_dim=d_model, **vit_cfg)

        # ResNet branch (nhận 4D hoặc 5D; nếu 5D sẽ pool theo T)
        self.resb = ResNet(in_ch=C_cdc, **resnet_cfg)

        # learnable fusion weight alpha in (0,1) via sigmoid
        self._alpha = nn.Parameter(torch.tensor(0.5))   # khởi tạo 0.5

        self.proj = nn.Linear(512,d_model,bias=True)
    @property
    def alpha(self):
        return torch.sigmoid(self._alpha)  # (0,1)

    def forward(self, x):
        x = x.to(self.device)

        fmap = self.cdcn(x)  # ảnh: B,C,H,W  | video: B,T,C,H,W

        vit_vec,_  = self.vit(fmap)               # B,D

        res_vec  = self.resb(fmap)
        B,T,C,H,W = res_vec.shape
        print(res_vec.shape)
        res_vec = F.adaptive_avg_pool2d(res_vec.view(B*T,C,H,W), 1).flatten(1)
        res_vec = res_vec.view(B,T,C).mean(dim=1)
        res_vec = self.proj(res_vec)


        a = self.alpha
        fused = F.normalize(a * vit_vec + (1 - a) * res_vec, dim=1)

        return {
            "vit": vit_vec,         # B,D
            "resnet": res_vec,      # B,D
            "fused": fused,         # B,D
            "alpha": a.detach()     # scalar tensor in (0,1)
        }


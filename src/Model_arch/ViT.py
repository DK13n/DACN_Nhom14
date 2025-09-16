import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalViT(nn.Module):

    def __init__(
        self,
        in_channels: int,      # C từ CDCN (ví dụ 128)
        embed_dim: int = 256,  # D
        depth: int = 4,        # số encoder blocks
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_spatial_proj: bool = False,   # nếu True: dùng conv 1x1 trước GAP
        max_len: int = 256,               # T tối đa để khởi tạo pos embedding
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_spatial_proj = use_spatial_proj

        # (1) Tùy chọn: ép kênh không gian trước khi GAP (giảm/đổi C)
        if use_spatial_proj:
            self.spatial_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            self.spatial_bn   = nn.BatchNorm2d(in_channels)
        else:
            self.spatial_proj = None

        # (2) Frame → vector: GAP: (B,T,C,H,W) -> (B,T,C)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # (3) Chiếu kênh C -> D
        self.frame_proj = nn.Linear(in_channels, embed_dim, bias=False)

        # (4) Positional + CLS (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))  # +1 cho CLS
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # (5) Transformer Encoder (theo thời gian)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,        # output: (B, S, D)
            activation='gelu',
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        B, T, C, H, W = x.shape
        assert C == self.in_channels, f"in_channels mismatch: {C} vs {self.in_channels}"

        # (1) xử lý từng frame theo không gian
        x = x.view(B*T, C, H, W)
        if self.spatial_proj is not None:
            x = self.spatial_bn(self.spatial_proj(x))
            x = F.relu(x, inplace=True)

        # GAP: B*T,C,1,1 -> B*T,C
        x = self.gap(x).view(B, T, C)

        # (2) chiếu C -> D
        x = self.frame_proj(x)                    # (B,T,D)

        # (3) thêm CLS và pos
        cls = self.cls_token.expand(B, -1, -1)    # (B,1,D)
        tokens = torch.cat([cls, x], dim=1)       # (B,T+1,D)

        # bảo vệ nếu T > max_len (cắt pos cho vừa)
        pos = self.pos_embed[:, : (T+1), :]       # (1,T+1,D)
        tokens = tokens + pos
        tokens = self.dropout(tokens)

        # (4) attention mask cho padding (tuỳ chọn)
        # PyTorch expects True = to-be-ignored. Ta cần mask có shape (B, S) với S=T+1.
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, T)
            # prepend False cho CLS (không mask CLS)
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device)
            enc_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)   # (B,T+1)
        else:
            enc_padding_mask = None

        # (5) Transformer Encoder theo thời gian
        z = self.encoder(tokens, src_key_padding_mask=enc_padding_mask)  # (B,T+1,D)
        z = self.norm(z)

        # (6) Tách cls và seq
        cls_out = z[:, 0, :]           # (B,D)
        seq_out = z[:, 1:, :]          # (B,T,D)

        return cls_out, seq_out



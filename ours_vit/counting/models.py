import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dinov3.hub import backbones as dino_backbones
except ImportError:
    dino_backbones = None


class DinoCountHead(nn.Module):
    """Lightweight density head: conv -> BN -> ReLU -> 1x1 conv + ReLU."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        density = self.decode(feats)
        density = F.softplus(density)
        density = F.interpolate(density, size=out_size, mode="bilinear", align_corners=False)
        return density


class DinoV3Density(nn.Module):
    """
    DINOv3 ViT backbone + multi-class density head.

    注意：
    - 不要在模型内部强制 no_grad()/冻结，否则外部 full finetune 会失效
    - 冻结/解冻通过训练脚本的 requires_grad 控制（或 init 的 freeze_backbone）
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        num_classes: int = 8,
        image_size: int = 448,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        freeze_backbone: bool = False,  # 改为默认不冻结，便于全参训练
    ):
        super().__init__()
        if dino_backbones is None:
            raise ImportError("Cannot import dinov3.hub.backbones - make sure dinov3 is in sys.path")

        print(f"Initializing {model_name} backbone from local dinov3...")
        self.backbone = getattr(dino_backbones, model_name)(pretrained=False)
        print(f"Created {model_name} backbone")

        if checkpoint_path:
            print(f"Loading backbone weights from {checkpoint_path}...")
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print("Loaded backbone checkpoint")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            if pretrained:
                print("Warning: pretrained=True but checkpoint_path=None; backbone is randomly initialized.")

        # 仅在 init 阶段按需冻结；forward 不再强制 no_grad
        if freeze_backbone:
            print("Freezing backbone parameters...")
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            print("Backbone is trainable (not frozen).")

        with torch.no_grad():
            dummy = torch.randn(1, 3, image_size, image_size)
            feats = self.backbone.forward_features(dummy)["x_norm_patchtokens"]
            embed_dim = feats.shape[-1]

        ps = self.backbone.patch_size
        if isinstance(ps, (tuple, list)):
            ph, pw = ps
        else:
            ph = pw = int(ps)
        self.patch_size = (ph, pw)

        print(f"Feature dimension: {embed_dim}")
        self.head = DinoCountHead(embed_dim, num_classes)
        self.num_classes = num_classes
        print(f"Density head initialized with {num_classes} channels")

    def _backbone_trainable(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, orig_h, orig_w = x.shape
        ph, pw = self.patch_size

        # pad 到 patch 对齐（避免 token 数不匹配）
        pad_h = (ph - orig_h % ph) % ph
        pad_w = (pw - orig_w % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        height = orig_h + pad_h
        width = orig_w + pad_w

        backbone_trainable = self._backbone_trainable()
        self.backbone.train(self.training and backbone_trainable)

        with torch.set_grad_enabled(self.training and backbone_trainable):
            out = self.backbone.forward_features(x)

        patch_tokens = out["x_norm_patchtokens"]  # [B, N, C]
        _, num_tokens, embed_dim = patch_tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != num_tokens:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch}, num_tokens={num_tokens}")

        spatial_features = patch_tokens.reshape(batch_size, h_patch, w_patch, embed_dim).permute(0, 3, 1, 2)
        density_map = self.head(spatial_features, (height, width))

        # 去掉 pad 区域，避免 pad 影响计数
        density_map = density_map[:, :, :orig_h, :orig_w]
        counts = density_map.flatten(2).sum(dim=2)  # [B, num_classes]
        return density_map, counts


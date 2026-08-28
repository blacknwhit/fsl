import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dinov3.hub import backbones as dino_backbones
except ImportError:
    dino_backbones = None


class DinoSegHead(nn.Module):
    """FCN-style linear head: conv + BN + ReLU + 1x1, then bilinear upsample."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        logits = self.decode(feats)
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits


class DinoV3Segmentation(nn.Module):
    """
    DINOv3 ViT backbone + FCN-style head.

    注意：
    - 不要在模型内部“强制冻结/强制 no_grad()”，否则外部想做 full finetune 会失效。
    - 冻结/解冻应该交给训练脚本通过 requires_grad 控制。
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        num_classes: int = 11,
        image_size: int = 448,
        pretrained: bool = True,  # 保留参数位以兼容，但当前走 checkpoint_path
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        print(f"Initializing {model_name} backbone from local dinov3...")

        if dino_backbones is None:
            raise ImportError("Cannot import dinov3.hub.backbones - make sure dinov3 is in sys.path")

        # 创建 backbone（不在这里冻结）
        self.backbone = getattr(dino_backbones, model_name)(pretrained=False)
        print(f"Created {model_name} backbone")

        # 加载自定义 checkpoint
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
            print("Warning: No checkpoint provided, using random initialization")

        # 推断 embed_dim（这里用 no_grad 即可）
        with torch.no_grad():
            dummy = torch.randn(1, 3, image_size, image_size)
            feats = self.backbone.forward_features(dummy)["x_norm_patchtokens"]
            embed_dim = feats.shape[-1]

        print(f"Feature dimension: {embed_dim}")
        self.head = DinoSegHead(embed_dim, num_classes)
        print(f"Segmentation head initialized with {num_classes} classes")

    def _backbone_trainable(self) -> bool:
        # 只要有任一参数 requires_grad=True，就认为 backbone 需要参与反传
        return any(p.requires_grad for p in self.backbone.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        backbone_trainable = self._backbone_trainable()

        # 训练时：如果 backbone 可训练，则 backbone 用 train() 并允许梯度；否则 eval()+no_grad
        # 验证/推理时：self.training=False，自然关闭梯度
        self.backbone.train(self.training and backbone_trainable)

        with torch.set_grad_enabled(self.training and backbone_trainable):
            out = self.backbone.forward_features(x)

        patch_embeddings = out["x_norm_patchtokens"]  # [B, N, C]
        B_p, N, C = patch_embeddings.shape

        patch_size = self.backbone.patch_size
        H_patch = H // patch_size
        W_patch = W // patch_size

        spatial_features = patch_embeddings.reshape(B_p, H_patch, W_patch, C).permute(0, 3, 1, 2)
        return self.head(spatial_features, (H, W))

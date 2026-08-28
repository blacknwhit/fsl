from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dinov3.hub import backbones as dino_backbones
except ImportError:
    dino_backbones = None

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from lora_multitask.lora import (
    LoRAConfig,
    count_trainable_params,
    inject_lora_into_dinov3_ffn,
    mark_only_lora_as_trainable,
)


class SharedDinoV3Backbone(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        *,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        if dino_backbones is None:
            raise ImportError("Cannot import dinov3.hub.backbones - make sure dinov3 is in sys.path")

        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.backbone = getattr(dino_backbones, model_name)(pretrained=False)
        self.use_lora = bool(use_lora)
        self.lora_config = (
            LoRAConfig(rank=int(lora_rank), alpha=float(lora_alpha), dropout=float(lora_dropout)) if self.use_lora else None
        )

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state, strict=False)

        with torch.no_grad():
            dummy = torch.randn(1, 3, image_size, image_size)
            tokens = self.backbone.forward_features(dummy)["x_norm_patchtokens"]
            self.embed_dim = int(tokens.shape[-1])

        ps = self.backbone.patch_size
        if isinstance(ps, (tuple, list)):
            ph, pw = int(ps[0]), int(ps[1])
        else:
            ph = pw = int(ps)
        self.patch_size = (ph, pw)

        if self.use_lora:
            replaced = inject_lora_into_dinov3_ffn(self.backbone, cfg=self.lora_config)  # type: ignore[arg-type]
            mark_only_lora_as_trainable(self.backbone)
            trainable = count_trainable_params(self.backbone)
            if trainable == 0:
                raise RuntimeError("LoRA enabled but no trainable params found after injection")
            print(f"[lora] enabled: replaced_linear={replaced}, trainable_params={trainable}")

    def export_config(self) -> dict:
        config = {
            "model_name": getattr(self.backbone, "__class__", type(self.backbone)).__name__.lower(),
            "image_size": int(getattr(self, "image_size", 448)),
            "use_lora": bool(self.use_lora),
            "lora_rank": 0,
            "lora_alpha": 0.0,
            "lora_dropout": 0.0,
        }
        if self.lora_config is not None:
            config["lora_rank"] = int(self.lora_config.rank)
            config["lora_alpha"] = float(self.lora_config.alpha)
            config["lora_dropout"] = float(self.lora_config.dropout)
        if hasattr(self, "model_name"):
            config["model_name"] = str(self.model_name)
        return config

    def _trainable(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def forward_features(self, x: torch.Tensor, *, trainable_override: bool | None = None) -> dict:
        if self.use_lora:
            trainable_override = None
        trainable = bool(trainable_override) if trainable_override is not None else self._trainable()
        self.backbone.train(self.training and trainable)
        with torch.set_grad_enabled(self.training and trainable):
            return self.backbone.forward_features(x)


class _DetBackboneAdapter(nn.Module):
    """
    Torchvision detector backbone adapter.
    Returns a single feature map tensor and exposes out_channels.
    """

    def __init__(self, shared: SharedDinoV3Backbone, out_channels: int = 256, *, trainable_backbone: bool = True):
        super().__init__()
        self.shared = shared
        self.proj = nn.Conv2d(shared.embed_dim, out_channels, kernel_size=1)
        self.out_channels = int(out_channels)
        self.trainable_backbone = bool(trainable_backbone)
        self.last_shared_representation: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        ph, pw = self.shared.patch_size

        pad_h = (ph - height % ph) % ph
        pad_w = (pw - width % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            height += pad_h
            width += pad_w

        tokens = self.shared.forward_features(x, trainable_override=self.trainable_backbone)["x_norm_patchtokens"]
        bsz, n, c = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != n:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch} vs N={n}")

        feat = tokens.reshape(bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)
        self.last_shared_representation = feat
        return self.proj(feat)


class DinoSegHead(nn.Module):
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
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class DinoCountHead(nn.Module):
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
        return F.interpolate(density, size=out_size, mode="bilinear", align_corners=False)


@dataclass(frozen=True)
class TaskOutputs:
    det_loss: torch.Tensor
    seg_loss: torch.Tensor
    cnt_loss: torch.Tensor


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        shared: SharedDinoV3Backbone,
        det_num_classes: int,
        seg_num_classes: int,
        cnt_num_classes: int,
        image_size: int = 448,
        det_out_channels: int = 256,
        det_image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        det_image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        det_train_backbone: bool = True,
        seg_train_backbone: bool = True,
        cnt_train_backbone: bool = True,
    ):
        super().__init__()
        self.shared = shared
        self.det_num_classes = int(det_num_classes)
        self.seg_num_classes = int(seg_num_classes)
        self.cnt_num_classes = int(cnt_num_classes)
        self.det_out_channels = int(det_out_channels)
        self.image_size = int(image_size)
        self.det_train_backbone = bool(det_train_backbone)
        self.seg_train_backbone = bool(seg_train_backbone)
        self.cnt_train_backbone = bool(cnt_train_backbone)

        det_backbone = _DetBackboneAdapter(shared, out_channels=det_out_channels, trainable_backbone=self.det_train_backbone)
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
        self.detector = FasterRCNN(
            det_backbone,
            num_classes=int(det_num_classes) + 1,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=image_size,
            max_size=image_size,
            image_mean=det_image_mean,
            image_std=det_image_std,
        )

        self.seg_head = DinoSegHead(shared.embed_dim, int(seg_num_classes))
        self.cnt_head = DinoCountHead(shared.embed_dim, int(cnt_num_classes))

    def shared_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.shared.backbone.parameters() if parameter.requires_grad]

    def det_task_specific_parameters(self) -> list[nn.Parameter]:
        shared_ids = {id(parameter) for parameter in self.shared.backbone.parameters()}
        return [
            parameter
            for parameter in self.detector.parameters()
            if parameter.requires_grad and id(parameter) not in shared_ids
        ]

    def seg_task_specific_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.seg_head.parameters() if parameter.requires_grad]

    def cnt_task_specific_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.cnt_head.parameters() if parameter.requires_grad]

    def task_specific_parameters(self) -> dict[str, list[nn.Parameter]]:
        return {
            "det": self.det_task_specific_parameters(),
            "seg": self.seg_task_specific_parameters(),
            "cnt": self.cnt_task_specific_parameters(),
        }

    def _tokens_to_feature_map(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        ph, pw = self.shared.patch_size
        bsz, n, c = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != n:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch}, num_tokens={n}")
        return tokens.reshape(bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)

    def forward_det(self, images, targets=None, *, return_representation: bool = False):
        outputs = self.detector(images, targets)
        if not return_representation:
            return outputs
        representation = self.detector.backbone.last_shared_representation
        if representation is None:
            raise RuntimeError("Detection shared representation was not captured.")
        return outputs, representation

    def forward_seg(self, x: torch.Tensor, *, return_representation: bool = False):
        _, _, h, w = x.shape
        tokens = self.shared.forward_features(x, trainable_override=self.seg_train_backbone)["x_norm_patchtokens"]
        feat = self._tokens_to_feature_map(tokens, h, w)
        logits = self.seg_head(feat, (h, w))
        if return_representation:
            return logits, feat
        return logits

    @staticmethod
    def _cnt_feat_with_scaled_backbone_grad(feat: torch.Tensor, mult: float) -> torch.Tensor:
        m = float(mult)
        if m == 1.0:
            return feat
        if not feat.requires_grad:
            return feat
        feat2 = feat.clone()
        feat2.register_hook(lambda g: g * m)
        return feat2

    def forward_cnt(
        self,
        x: torch.Tensor,
        *,
        cnt_backbone_grad_mult: float = 1.0,
        return_representation: bool = False,
    ):
        _, _, orig_h, orig_w = x.shape
        ph, pw = self.shared.patch_size
        pad_h = (ph - orig_h % ph) % ph
        pad_w = (pw - orig_w % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        height = orig_h + pad_h
        width = orig_w + pad_w

        tokens = self.shared.forward_features(x, trainable_override=self.cnt_train_backbone)["x_norm_patchtokens"]
        feat = self._tokens_to_feature_map(tokens, height, width)
        feat = self._cnt_feat_with_scaled_backbone_grad(feat, cnt_backbone_grad_mult)

        density = self.cnt_head(feat, (height, width))
        density = density[:, :, :orig_h, :orig_w]
        counts = density.flatten(2).sum(dim=2)
        if return_representation:
            return density, counts, feat
        return density, counts

    def forward(self, mode: str, *args, **kwargs):
        mode = str(mode).lower()
        if mode == "det":
            return self.forward_det(*args, **kwargs)
        if mode == "seg":
            return self.forward_seg(*args, **kwargs)
        if mode == "cnt":
            return self.forward_cnt(*args, **kwargs)
        raise ValueError(f"Unknown forward mode: {mode}")

    def export_config(self) -> dict:
        config = self.shared.export_config()
        config.update(
            {
                "det_num_classes": self.det_num_classes,
                "seg_num_classes": self.seg_num_classes,
                "cnt_num_classes": self.cnt_num_classes,
                "det_out_channels": self.det_out_channels,
                "image_size": self.image_size,
                "det_train_backbone": bool(self.det_train_backbone),
                "seg_train_backbone": bool(self.seg_train_backbone),
                "cnt_train_backbone": bool(self.cnt_train_backbone),
            }
        )
        return config

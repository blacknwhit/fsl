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

from .lora import LoRAConfig, count_trainable_params, inject_lora_into_dinov3_ffn, mark_only_lora_as_trainable


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

    def _trainable(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        trainable_override: bool | None = None,
    ) -> dict:
        if self.use_lora:
            # LoRA mode always computes gradients through the backbone during training;
            # per-task freeze flags are ignored (only LoRA params are trainable anyway).
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

    def __init__(
        self,
        shared: SharedDinoV3Backbone,
        out_channels: int = 256,
        *,
        trainable_backbone: bool = True,
    ):
        super().__init__()
        self.shared = shared
        self.proj = nn.Conv2d(shared.embed_dim, out_channels, kernel_size=1)
        self.out_channels = int(out_channels)
        self.trainable_backbone = bool(trainable_backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        ph, pw = self.shared.patch_size

        pad_h = (ph - height % ph) % ph
        pad_w = (pw - width % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            height += pad_h
            width += pad_w

        tokens = self.shared.forward_features(
            x,
            trainable_override=self.trainable_backbone,
        )["x_norm_patchtokens"]  # [B,N,C]
        bsz, n, c = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != n:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch} vs N={n}")

        feat = tokens.reshape(bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)
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
        self.det_train_backbone = bool(det_train_backbone)
        self.seg_train_backbone = bool(seg_train_backbone)
        self.cnt_train_backbone = bool(cnt_train_backbone)

        det_backbone = _DetBackboneAdapter(
            shared,
            out_channels=det_out_channels,
            trainable_backbone=self.det_train_backbone,
        )
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
        self.cnt_num_classes = int(cnt_num_classes)

    def forward_det(self, images, targets=None):
        return self.detector(images, targets)

    def forward_seg(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        ph, pw = self.shared.patch_size
        tokens = self.shared.forward_features(
            x,
            trainable_override=self.seg_train_backbone,
        )["x_norm_patchtokens"]
        _, n, c = tokens.shape
        h_patch = h // ph
        w_patch = w // pw
        feat = tokens.reshape(bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)
        return self.seg_head(feat, (h, w))

    @staticmethod
    def _cnt_feat_with_scaled_backbone_grad(feat: torch.Tensor, mult: float) -> torch.Tensor:
        """
        方式A：只缩放“计数分支 -> backbone”的梯度，不影响计数头参数的梯度。
        做法：对输入给 cnt_head 的 feat 做 clone，并在该 clone 上 register_hook 缩放 dL/dfeat。
        """
        m = float(mult)
        if m == 1.0:
            return feat
        # 只有在需要反传时才能挂 hook（否则会报错）
        if not feat.requires_grad:
            return feat
        feat2 = feat.clone()
        feat2.register_hook(lambda g: g * m)
        return feat2

    def forward_cnt(self, x: torch.Tensor, *, cnt_backbone_grad_mult: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, orig_h, orig_w = x.shape
        ph, pw = self.shared.patch_size
        pad_h = (ph - orig_h % ph) % ph
        pad_w = (pw - orig_w % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        height = orig_h + pad_h
        width = orig_w + pad_w

        tokens = self.shared.forward_features(
            x,
            trainable_override=self.cnt_train_backbone,
        )["x_norm_patchtokens"]
        _, n, c = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != n:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch}, num_tokens={n}")
        feat = tokens.reshape(bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)

        # 只对“计数任务回传到backbone”的梯度做缩放；不影响cnt_head参数梯度
        feat = self._cnt_feat_with_scaled_backbone_grad(feat, cnt_backbone_grad_mult)

        density = self.cnt_head(feat, (height, width))
        density = density[:, :, :orig_h, :orig_w]
        counts = density.flatten(2).sum(dim=2)  # [B,C]
        return density, counts

    def forward_seg_and_cnt(
        self,
        seg_x: torch.Tensor,
        cnt_x: torch.Tensor,
        *,
        cnt_backbone_grad_mult: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fused seg+cnt forward that runs the shared backbone once.

        Requirements (to keep behavior consistent with separate forwards):
        - seg_x and cnt_x must have the same [B,3,H,W] spatial shape.
        - H/W must already be patch-aligned (no implicit padding), otherwise segmentation behavior would change.
        - seg_train_backbone and cnt_train_backbone must match (train/eval mode of the shared backbone).
        """
        if self.seg_train_backbone != self.cnt_train_backbone:
            raise ValueError("forward_seg_and_cnt requires seg_train_backbone == cnt_train_backbone")

        if seg_x.dim() != 4 or cnt_x.dim() != 4:
            raise ValueError("Expected seg_x/cnt_x to be [B,3,H,W] tensors")
        if seg_x.shape[1:] != cnt_x.shape[1:]:
            raise ValueError(f"seg_x and cnt_x must have same C/H/W, got {tuple(seg_x.shape)} vs {tuple(cnt_x.shape)}")

        seg_bsz, _, h, w = seg_x.shape
        cnt_bsz = int(cnt_x.shape[0])
        ph, pw = self.shared.patch_size
        if (h % ph) or (w % pw):
            raise ValueError(f"Fused seg+cnt requires patch-aligned H/W, got {(h, w)} for patch {self.shared.patch_size}")

        x = torch.cat([seg_x, cnt_x], dim=0)
        tokens = self.shared.forward_features(x, trainable_override=self.seg_train_backbone)["x_norm_patchtokens"]
        _, n, c = tokens.shape
        h_patch = h // ph
        w_patch = w // pw
        if h_patch * w_patch != n:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch}, num_tokens={n}")

        feat = tokens.reshape(seg_bsz + cnt_bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)
        seg_feat = feat[:seg_bsz]
        cnt_feat = feat[seg_bsz:]

        seg_logits = self.seg_head(seg_feat, (h, w))

        # 关键：只缩放 cnt 分支回传到 backbone 的梯度（seg 不受影响）
        cnt_feat = self._cnt_feat_with_scaled_backbone_grad(cnt_feat, cnt_backbone_grad_mult)

        density = self.cnt_head(cnt_feat, (h, w))
        counts = density.flatten(2).sum(dim=2)
        return seg_logits, density, counts

    def forward(self, task: str, *args, **kwargs):
        task_key = str(task).lower()
        if task_key == "det":
            return self.forward_det(*args, **kwargs)
        if task_key == "seg":
            return self.forward_seg(*args, **kwargs)
        if task_key == "cnt":
            return self.forward_cnt(*args, **kwargs)
        if task_key in {"seg_cnt", "seg+cnt"}:
            return self.forward_seg_and_cnt(*args, **kwargs)
        raise ValueError(f"Unsupported task '{task}'. Expected one of: det, seg, cnt, seg_cnt")

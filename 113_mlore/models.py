from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

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

try:
    from .mlore import MLoREDecoder
except ImportError:
    from mlore import MLoREDecoder


@dataclass(frozen=True)
class TaskFeatureMeta:
    original_size: tuple[int, int]
    padded_size: tuple[int, int]
    pad_h: int
    pad_w: int


class SharedDinoV3Backbone(nn.Module):
    TASK_NAMES = ("det", "seg", "cnt")

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        *,
        mlore_decoder_dim: int = 256,
        mlore_rank_list: Sequence[int] = (8, 16, 24, 32, 40, 48),
        mlore_topk: int = 4,
        mlore_task_rank: int = 32,
        mlore_pre_softmax: bool = False,
        mlore_load_balancing_weight: float = 3e-4,
        mlore_select_layers: Sequence[int] = (23,),
        mlore_num_stages: int = 1,
        grad_checkpointing: bool = True,
    ):
        super().__init__()
        if dino_backbones is None:
            raise ImportError("Cannot import dinov3.hub.backbones - make sure dinov3 is available")
        if model_name != "dinov3_vitl16":
            raise ValueError("113_mlore currently supports only dinov3_vitl16.")

        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.selected_layers = tuple(int(index) for index in mlore_select_layers)
        self.decoder_dim = int(mlore_decoder_dim)
        self.rank_list = tuple(int(rank) for rank in mlore_rank_list)
        self.topk = int(mlore_topk)
        self.task_rank = int(mlore_task_rank)
        self.pre_softmax = bool(mlore_pre_softmax)
        self.load_balancing_weight = float(mlore_load_balancing_weight)
        self.num_stages = int(mlore_num_stages)
        self.grad_checkpointing = bool(grad_checkpointing)

        self.backbone = getattr(dino_backbones, self.model_name)(pretrained=False)
        if checkpoint_path:
            try:
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(checkpoint_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            tokens = self.backbone.forward_features(dummy)["x_norm_patchtokens"]
            self.embed_dim = int(tokens.shape[-1])

        ps = self.backbone.patch_size
        if isinstance(ps, (tuple, list)):
            self.patch_size = (int(ps[0]), int(ps[1]))
        else:
            self.patch_size = (int(ps), int(ps))

        if not self.selected_layers:
            raise ValueError("mlore_select_layers must not be empty.")
        if max(self.selected_layers) >= len(self.backbone.blocks):
            raise ValueError(
                f"mlore_select_layers {self.selected_layers} exceed backbone depth {len(self.backbone.blocks)}"
            )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.decoder = MLoREDecoder(
            embed_dim=self.embed_dim,
            decoder_dim=self.decoder_dim,
            tasks=self.TASK_NAMES,
            num_layers=len(self.selected_layers),
            num_stages=self.num_stages,
            rank_list=self.rank_list,
            task_rank=self.task_rank,
            topk=self.topk,
            pre_softmax=self.pre_softmax,
            grad_checkpointing=self.grad_checkpointing,
            im_size=(self.image_size // self.patch_size[0]) * (self.image_size // self.patch_size[1]),
        )
        self._last_lb_loss: torch.Tensor | None = None

    def export_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "image_size": self.image_size,
            "mlore_decoder_dim": self.decoder_dim,
            "mlore_rank_list": list(self.rank_list),
            "mlore_topk": self.topk,
            "mlore_task_rank": self.task_rank,
            "mlore_pre_softmax": self.pre_softmax,
            "mlore_load_balancing_weight": self.load_balancing_weight,
            "mlore_select_layers": list(self.selected_layers),
            "mlore_num_stages": self.num_stages,
            "grad_checkpointing": self.grad_checkpointing,
        }

    def pop_load_balancing_loss(self) -> torch.Tensor:
        device = next(self.decoder.parameters()).device
        if self._last_lb_loss is None or self.load_balancing_weight <= 0:
            self._last_lb_loss = None
            return torch.zeros((), device=device)
        loss = self._last_lb_loss
        self._last_lb_loss = None
        return loss

    def _pad_to_patch(self, x: torch.Tensor) -> tuple[torch.Tensor, TaskFeatureMeta]:
        _, _, height, width = x.shape
        ph, pw = self.patch_size
        pad_h = (ph - height % ph) % ph
        pad_w = (pw - width % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        meta = TaskFeatureMeta(
            original_size=(height, width),
            padded_size=(height + pad_h, width + pad_w),
            pad_h=int(pad_h),
            pad_w=int(pad_w),
        )
        return x, meta

    def _apply_block(self, block: nn.Module, x: torch.Tensor, rope) -> torch.Tensor:
        try:
            return block(x, rope=rope)
        except TypeError:
            return block(x)

    def _extract_multi_scale_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        self.backbone.eval()
        x, (height_tokens, width_tokens) = self.backbone.prepare_tokens_with_masks(x, masks=None)

        rope = None
        if hasattr(self.backbone, "rope_embed") and self.backbone.rope_embed is not None:
            rope = self.backbone.rope_embed(H=height_tokens, W=width_tokens)

        n_prefix = 1 + int(getattr(self.backbone, "n_storage_tokens", 0))
        last_block_index = len(self.backbone.blocks) - 1
        selected_set = set(self.selected_layers)
        collected: Dict[int, torch.Tensor] = {}

        for index, block in enumerate(self.backbone.blocks):
            x = self._apply_block(block, x, rope)
            if index in selected_set and index != last_block_index:
                collected[index] = x[:, n_prefix:]

        if last_block_index in selected_set:
            if getattr(self.backbone, "untie_cls_and_patch_norms", False):
                collected[last_block_index] = self.backbone.norm(x[:, n_prefix:])
            else:
                collected[last_block_index] = self.backbone.norm(x)[:, n_prefix:]

        feature_maps = []
        for index in self.selected_layers:
            tokens = collected[index]
            bsz, num_tokens, channels = tokens.shape
            if height_tokens * width_tokens != num_tokens:
                raise ValueError(
                    f"Token mismatch at layer {index}: expected {height_tokens * width_tokens}, got {num_tokens}"
                )
            feature_maps.append(tokens.reshape(bsz, height_tokens, width_tokens, channels).permute(0, 3, 1, 2))
        return feature_maps

    def _extract_task_inputs(self, x: torch.Tensor) -> tuple[list[torch.Tensor], TaskFeatureMeta]:
        x, meta = self._pad_to_patch(x)
        with torch.no_grad():
            multi_scale_features = self._extract_multi_scale_features(x)
        return multi_scale_features, meta

    def forward_all_task_features(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], TaskFeatureMeta]:
        multi_scale_features, meta = self._extract_task_inputs(x)
        feature_maps = {}
        lb_losses = []
        for task_name in self.TASK_NAMES:
            feature_map, lb_loss = self.decoder.forward_task_features(multi_scale_features, task_name)
            feature_maps[task_name] = feature_map
            lb_losses.append(lb_loss)
        if lb_losses and self.load_balancing_weight > 0:
            self._last_lb_loss = torch.stack(lb_losses).mean() * self.load_balancing_weight
        else:
            self._last_lb_loss = None
        return feature_maps, meta

    def forward_task_features(self, x: torch.Tensor, task_name: str) -> tuple[torch.Tensor, TaskFeatureMeta]:
        if task_name not in self.TASK_NAMES:
            raise ValueError(f"Unknown task_name: {task_name}")
        multi_scale_features, meta = self._extract_task_inputs(x)
        feature_map, lb_loss = self.decoder.forward_task_features(multi_scale_features, task_name)
        if self.load_balancing_weight > 0:
            self._last_lb_loss = lb_loss * self.load_balancing_weight
        else:
            self._last_lb_loss = None
        return feature_map, meta


class _DetBackboneAdapter(nn.Module):
    def __init__(self, shared: SharedDinoV3Backbone, out_channels: int = 256):
        super().__init__()
        self.shared = shared
        self.out_channels = int(out_channels)
        if int(shared.decoder_dim) == self.out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv2d(shared.decoder_dim, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map, _ = self.shared.forward_task_features(x, task_name="det")
        return self.proj(feature_map)


class DinoSegHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
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
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        density = self.decode(feats)
        density = F.softplus(density)
        return F.interpolate(density, size=out_size, mode="bilinear", align_corners=False)


class MultiTaskModel(nn.Module):
    TASK_ID_DET = "det"
    TASK_ID_SEG = "seg"
    TASK_ID_CNT = "cnt"

    def __init__(
        self,
        *,
        shared: SharedDinoV3Backbone,
        det_num_classes: int,
        seg_num_classes: int,
        cnt_num_classes: int,
        image_size: int = 448,
        det_out_channels: int = 256,
        det_image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        det_image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.shared = shared
        self.image_size = int(image_size)
        self.det_out_channels = int(det_out_channels)
        self.det_num_classes = int(det_num_classes)
        self.seg_num_classes = int(seg_num_classes)
        self.cnt_num_classes = int(cnt_num_classes)

        det_backbone = _DetBackboneAdapter(shared, out_channels=self.det_out_channels)
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
        self.detector = FasterRCNN(
            det_backbone,
            num_classes=self.det_num_classes + 1,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=self.image_size,
            max_size=self.image_size,
            image_mean=det_image_mean,
            image_std=det_image_std,
        )

        self.seg_head = DinoSegHead(shared.decoder_dim, self.seg_num_classes)
        self.cnt_head = DinoCountHead(shared.decoder_dim, self.cnt_num_classes)

    def export_config(self) -> dict:
        config = self.shared.export_config()
        config.update(
            {
                "det_num_classes": self.det_num_classes,
                "seg_num_classes": self.seg_num_classes,
                "cnt_num_classes": self.cnt_num_classes,
                "det_out_channels": self.det_out_channels,
            }
        )
        return config

    def pop_load_balancing_loss(self) -> torch.Tensor:
        return self.shared.pop_load_balancing_loss()

    def forward_det(self, images, targets=None, *, return_lb: bool = False):
        output = self.detector(images, targets)
        lb_loss = self.shared.pop_load_balancing_loss()
        if return_lb:
            return output, lb_loss
        return output

    def forward_seg(self, x: torch.Tensor, *, return_lb: bool = False):
        feature_map, meta = self.shared.forward_task_features(x, task_name=self.TASK_ID_SEG)
        logits = self.seg_head(feature_map, meta.padded_size)
        logits = logits[:, :, : meta.original_size[0], : meta.original_size[1]]
        lb_loss = self.shared.pop_load_balancing_loss()
        if return_lb:
            return logits, lb_loss
        return logits

    def forward_cnt(self, x: torch.Tensor, *, return_lb: bool = False) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_map, meta = self.shared.forward_task_features(x, task_name=self.TASK_ID_CNT)
        density = self.cnt_head(feature_map, meta.padded_size)
        density = density[:, :, : meta.original_size[0], : meta.original_size[1]]
        counts = density.flatten(2).sum(dim=2)
        lb_loss = self.shared.pop_load_balancing_loss()
        if return_lb:
            return density, counts, lb_loss
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

"""DINOv3 multitask baseline with LoRA and selective task-group updates."""

from .models import MultiTaskModel, SharedDinoV3Backbone

__all__ = ["MultiTaskModel", "SharedDinoV3Backbone"]

"""DINOv3 multitask baseline with LoRA and ConsMTL."""

from .models import MultiTaskModel, SharedDinoV3Backbone

__all__ = ["MultiTaskModel", "SharedDinoV3Backbone"]

from __future__ import annotations

try:
    from .mae_moe_wrapper import MAELoRAMoEBlockWrapper
except ImportError:
    from mae_moe_wrapper import MAELoRAMoEBlockWrapper


CLIPLoRAMoEBlockWrapper = MAELoRAMoEBlockWrapper

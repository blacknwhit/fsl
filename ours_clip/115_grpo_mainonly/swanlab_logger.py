from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass
class MetricLogger:
    enabled: bool = False

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        return

    def update_config(self, config: Mapping[str, Any]) -> None:
        return

    def finish(self) -> None:
        return


class SwanLabMetricLogger(MetricLogger):
    def __init__(
        self,
        *,
        enabled: bool,
        is_main_process: bool,
        project: str | None,
        workspace: str | None,
        experiment_name: str | None,
        mode: str | None,
        logdir: str | None,
        config: Mapping[str, Any] | None,
    ) -> None:
        super().__init__(enabled=bool(enabled) and bool(is_main_process))
        self._swanlab = None

        if not bool(enabled):
            return

        try:
            import swanlab  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "SwanLab logging is enabled but `swanlab` is not installed. "
                "Install it with `pip install swanlab`."
            ) from exc

        self._swanlab = swanlab
        if not bool(is_main_process):
            return

        init_kwargs: dict[str, Any] = {
            "config": self._sanitize_mapping(config or {}),
        }
        if project:
            init_kwargs["project"] = project
        if workspace:
            init_kwargs["workspace"] = workspace
        if experiment_name:
            init_kwargs["experiment_name"] = experiment_name
        if mode:
            init_kwargs["mode"] = mode
        if logdir:
            init_kwargs["logdir"] = logdir

        swanlab.init(**init_kwargs)

    def _sanitize_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value) if math.isfinite(value) else None
        if hasattr(value, "item"):
            try:
                return self._sanitize_value(value.item())
            except Exception:
                return str(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return str(list(value))
        if isinstance(value, dict):
            return str(value)
        return value if isinstance(value, str) else str(value)

    def _sanitize_mapping(self, data: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in data.items():
            clean = self._sanitize_value(value)
            if clean is None:
                continue
            out[str(key)] = clean
        return out

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        if not self.enabled or self._swanlab is None:
            return
        payload = self._sanitize_mapping(metrics)
        if not payload:
            return
        if step is None:
            self._swanlab.log(payload)
        else:
            self._swanlab.log(payload, step=int(step))

    def update_config(self, config: Mapping[str, Any]) -> None:
        if not self.enabled or self._swanlab is None:
            return
        payload = self._sanitize_mapping(config)
        if payload:
            self._swanlab.config.update(payload)

    def finish(self) -> None:
        if self.enabled and self._swanlab is not None:
            self._swanlab.finish()


def create_metric_logger(
    *,
    args,
    is_main_process: bool,
    save_dir: Path,
) -> MetricLogger:
    if not bool(getattr(args, "use_swanlab", False)):
        return MetricLogger()

    return SwanLabMetricLogger(
        enabled=True,
        is_main_process=is_main_process,
        project=(getattr(args, "swanlab_project", None) or Path.cwd().name),
        workspace=getattr(args, "swanlab_workspace", None),
        experiment_name=(getattr(args, "swanlab_experiment_name", None) or save_dir.name),
        mode=getattr(args, "swanlab_mode", None),
        logdir=(getattr(args, "swanlab_logdir", None) or str(save_dir / "swanlog")),
        config=vars(args),
    )

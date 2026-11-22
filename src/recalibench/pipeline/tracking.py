"""Optional MLflow logging (off by default; `mlflow.enabled: true` to turn on).

MLflow is not a dependency: it is imported only when enabled, and a clear
error says to install it if it isn't there.
"""

import importlib
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def log_metrics(cfg: DictConfig, result: dict[str, Any]) -> None:
    if not bool(OmegaConf.select(cfg, "mlflow.enabled", default=False)):
        return
    try:
        mlflow = cast(Any, importlib.import_module("mlflow"))
    except ImportError as e:
        raise RuntimeError(
            "mlflow.enabled is true but mlflow isn't installed: uv add mlflow"
        ) from e
    run_name = f"{result['task']}/{result['model_id']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {"task": result["task"], "model": result["model_id"], "selected": result["selected"]}
        )
        for side in ("raw", "calibrated"):
            for name, ci in result[side].items():
                mlflow.log_metric(f"{side}_{name}", float(ci["value"]))

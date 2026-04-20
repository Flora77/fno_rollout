import inspect
from typing import Any, Dict

import torch.nn as nn

try:
    from neuralop.models.fno import FNO, TFNO
except ImportError as exc:
    raise ImportError("Please install neuralop before running this project.") from exc



def _filter_model_kwargs(model_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(model_cls)
    return {k: v for k, v in kwargs.items() if v is not None and k in sig.parameters}



def build_model(cfg: Dict[str, Any]) -> nn.Module:
    arch = str(cfg["model"]["arch"]).lower()
    if arch == "fno":
        model_cls = FNO
    elif arch == "tfno":
        model_cls = TFNO
    else:
        raise ValueError(f"Unsupported model arch: {cfg['model']['arch']}")

    kwargs = dict(
        n_modes=tuple(cfg["model"]["n_modes"]),
        hidden_channels=int(cfg["model"]["hidden_channels"]),
        in_channels=int(cfg["data"]["input_steps"]),
        out_channels=int(cfg["data"]["output_steps"]),
        n_layers=int(cfg["model"]["n_layers"]),
        lifting_channels=int(cfg["model"]["lifting_channels"]),
        projection_channels=int(cfg["model"]["projection_channels"]),
    )
    return model_cls(**_filter_model_kwargs(model_cls, kwargs))

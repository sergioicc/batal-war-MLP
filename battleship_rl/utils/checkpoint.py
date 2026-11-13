"""Utilidades de guardado y carga de checkpoints y métricas (compatibles con Google Drive)."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch

from battleship_rl.config import Config


def resolve_checkpoint_dir(
    local_dir: str = "checkpoints",
    drive_dir: str = "/content/drive/MyDrive/battleship_rl/checkpoints",
) -> Path:
    """Usa Google Drive si está montado en Colab; en caso contrario usa carpeta local."""
    drive_root = Path("/content/drive/MyDrive")
    target = Path(drive_dir) if drive_root.exists() else Path(local_dir)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_checkpoint(
    filepath: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    block_idx: int,
    mu_ema: Optional[float],
    best_eval_mean: Optional[float],
    config: Config,
) -> Path:
    """Guarda pesos del MLP, estado del optimizador único, mu_ema y metadatos."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "block_idx": int(block_idx),
        "mu_ema": mu_ema,
        "best_eval_mean": best_eval_mean,
        "config": asdict(config),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    filepath: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Carga un checkpoint guardado previamente y restaura pesos y optimizador."""
    payload = torch.load(Path(filepath), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def save_metrics_csv(rows: List[Dict[str, Any]], filepath: Union[str, Path]) -> Path:
    """Guarda una lista de diccionarios de métricas en un archivo CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path

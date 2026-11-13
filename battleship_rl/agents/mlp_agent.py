"""Agente MLP único en PyTorch para Batalla Naval (501 entradas -> 100 logits)."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from battleship_rl.config import Config


def _build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"Función de activación no soportada: {name}")


class MLPPolicyNet(nn.Module):
    """Red neuronal Perceptrón Multicapa de 4 neuronas de entrada por casilla.

    Recibe la matriz (B, 100, 4) y aplica los mismos pesos a las 100 casillas
    para producir 100 logits (B, 100).
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (64, 32),
        output_dim: int = 1,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        in_features = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(_build_activation(activation))
            in_features = h_dim

        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

        # Inicialización ortogonal en la capa de salida (gain=0.02, 2x respecto a 0.01)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        obs: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mapea (B, 100, 4) -> logits (B, 100), enmascarando casillas ya atacadas con -1e9."""
        logits = self.net(obs).squeeze(-1)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -1e9)
        return logits


class MLPAgent:
    """Envoltorio de conveniencia del MLP para muestreo, evaluación y manejo de dispositivo."""

    def __init__(self, config: Config, device: Optional[torch.device] = None) -> None:
        self.config = config
        if device is None:
            if config.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(config.device)
        else:
            self.device = device

        self.policy = MLPPolicyNet(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
        ).to(self.device)

    def select_actions(
        self,
        obs_np: np.ndarray,
        valid_mask_np: np.ndarray,
        greedy: bool = False,
        with_grad: bool = True,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Selecciona una acción por partida en el lote.

        Returns:
            actions_np: ndarray int64 de forma (B,)
            log_probs: tensor (B,) con log pi(a|s)
            entropies: tensor (B,) con entropía de la distribución categórica
        """
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(valid_mask_np, dtype=torch.bool, device=self.device)

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            logits = self.policy(obs_t, mask_t)
            dist = Categorical(logits=logits)
            if greedy:
                actions_t = torch.argmax(logits, dim=-1)
            else:
                actions_t = dist.sample()
            log_probs = dist.log_prob(actions_t)
            entropies = dist.entropy()

        return actions_t.detach().cpu().numpy().astype(np.int64), log_probs, entropies

    def evaluate_actions(
        self,
        obs_t: torch.Tensor,
        valid_mask_t: torch.Tensor,
        actions_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-evalúa log_probs y entropía para un lote de transiciones almacenadas."""
        logits = self.policy(obs_t, valid_mask_t)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions_t), dist.entropy()

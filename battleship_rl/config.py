"""Configuración centralizada de hiperparámetros para entrenamiento, entorno y evaluación."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Config:
    # --- Semillas y dispositivo ---
    seed: int = 42
    eval_seed_base: int = 10_000
    device: str = "auto"  # "auto", "cuda", "cpu"

    # --- Entorno y bloques ---
    block_size: int = 20          # Rondas/partidas por bloque (vectorizado en lote de 20)
    num_blocks: int = 250         # Cantidad total de bloques de entrenamiento (250 * 20 = 5000 partidas)

    # --- Arquitectura MLP por casilla (matriz (100, 4) -> 4 neuronas entrada -> 1 logit) ---
    input_dim: int = 4
    hidden_dims: Tuple[int, ...] = (64, 32)
    output_dim: int = 1
    activation: str = "relu"      # "relu", "gelu", "tanh"
    use_layer_norm: bool = True

    # --- Recompensas inmediatas (por disparo, x2 respecto al ajuste previo) ---
    # SUNK comparte exactamente el mismo valor que HIT (sin bono adicional).
    reward_hit: float = 0.030
    reward_miss: float = -0.006

    # --- Retroalimentación por bloque (cada 20 rondas) ---
    # Señal macro:
    #   mean_term = w_mean * (mu_ema - mu_actual)
    #   factor_var = 1 / (1 + w_var * sigma_actual^2)   en (0, 1]
    #   R_block = mean_term * factor_var   (nunca cambia de signo)
    ema_alpha: float = 0.20             # Factor de actualización de la media móvil histórica mu_ema
    block_mean_weight: float = 1.20     # Refuerzo macro por mejora de media respecto a mu_ema
    block_var_weight: float = 0.05      # Atenuación multiplicativa por varianza (reduce |R_block| sin volverlo negativo)
    round_relative_weight: float = 0.0  # 0.0 = toda la señal del bloque conserva el signo de R_block
    clip_block_advantage: float = 5.0   # Límite de seguridad para evitar gradientes explosivos
    skip_first_block_mean_diff: bool = True  # En el bloque 1 inicializa mu_ema = mu_actual

    # --- Optimización (un único learning_rate y pérdida sencilla -(recompensa * log_prob)) ---
    learning_rate: float = 0.1          # Tasa de aprendizaje única (por disparo y por bloque)
    entropy_coef: float = 0.0           # 0.0 = pérdida pura -(recompensa * log_prob)
    block_update_epochs: int = 1        # Épocas de actualización macro sobre las 20 rondas
    max_grad_norm: float = 1.0          # Recorte de norma de gradiente

    # --- Evaluación periódica contra baselines ---
    log_every_blocks: int = 1
    eval_every_blocks: int = 10
    eval_num_games: int = 100
    eval_greedy: bool = False           # False = muestreo estocástico; True = argmax codicioso

    # --- Persistencia / Google Drive ---
    checkpoint_dir: str = "checkpoints"
    save_every_blocks: int = 25
    metrics_filename: str = "training_metrics.csv"
    eval_filename: str = "eval_metrics.csv"
    checkpoint_filename: str = "latest_checkpoint.pt"
    best_checkpoint_filename: str = "best_checkpoint.pt"

    # Metadatos informativos de experimento
    experiment_name: str = "mlp_reinforce_dual"
    notes: Tuple[str, ...] = field(default_factory=tuple)

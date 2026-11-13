"""Cálculo de estadísticas de bloque, señal macro de refuerzo y comparativas contra baselines."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from battleship_rl.config import Config
from battleship_rl.game.constants import NUM_CELLS
from battleship_rl.game.engine_ref import generate_random_fleet
from battleship_rl.game.engine_vec import BattleshipVecEnv

if TYPE_CHECKING:
    from battleship_rl.agents.mlp_agent import MLPAgent


def compute_block_stats(shots_per_round: np.ndarray) -> Dict[str, float]:
    """Calcula media, desviación estándar, mínimo y máximo de disparos en un bloque."""
    arr = np.asarray(shots_per_round, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("shots_per_round debe ser un arreglo 1D no vacío.")
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_rounds": float(arr.size),
    }


def compute_block_feedback(
    mu_actual: float,
    sigma_actual: float,
    mu_ema: Optional[float],
    config: Config,
) -> Tuple[float, float, float, float]:
    """Calcula la señal macro del bloque de 20 rondas y actualiza mu_ema.

    - Comparación de promedio: usa `mu_ema` como referencia (`mean_term > 0` si el
      promedio bajó; `< 0` si subió).
    - Atenuación por varianza: multiplica `mean_term` por `1 / (1 + w_var * var)`
      en `(0, 1]`. Una varianza alta reduce la magnitud `|block_signal|` sin
      cambiar nunca su signo (si el promedio bajó, sigue siendo positivo).

    Returns:
        (block_signal, next_mu_ema, mean_term, var_factor)
    """
    if mu_ema is None:
        mean_term = 0.0
        next_mu_ema = float(mu_actual)
    else:
        mean_term = config.block_mean_weight * (float(mu_ema) - float(mu_actual))
        next_mu_ema = (1.0 - config.ema_alpha) * float(mu_ema) + config.ema_alpha * float(mu_actual)

    var_actual = float(sigma_actual) ** 2
    var_factor = 1.0 / (1.0 + config.block_var_weight * var_actual)
    block_signal = float(mean_term * var_factor)
    return block_signal, next_mu_ema, float(mean_term), float(var_factor)


def generate_eval_fleets(num_games: int, seed: int) -> np.ndarray:
    """Genera un lote reproducible de flotas (num_games, 100) para evaluación justa."""
    rng = np.random.default_rng(seed)
    fleets = np.empty((num_games, NUM_CELLS), dtype=np.int8)
    for i in range(num_games):
        fleets[i] = generate_random_fleet(rng)
    return fleets


def evaluate_mlp_agent(
    agent: "MLPAgent",
    fixed_ship_ids: np.ndarray,
    greedy: bool = False,
) -> np.ndarray:
    """Evalúa el agente MLP sobre un lote fijo de flotas sin actualizar pesos."""
    n_games = fixed_ship_ids.shape[0]
    env = BattleshipVecEnv(batch_size=n_games)
    obs_np = env.reset(fixed_ship_ids=fixed_ship_ids)

    was_training = agent.policy.training
    agent.policy.eval()
    try:
        for _ in range(NUM_CELLS):
            if np.all(env.done):
                break
            active_idx = np.flatnonzero(~env.done)
            valid_mask_np = env.valid_mask()
            active_actions, _, _ = agent.select_actions(
                obs_np[active_idx],
                valid_mask_np[active_idx],
                greedy=greedy,
                with_grad=False,
            )
            actions_full = np.zeros((n_games,), dtype=np.int64)
            actions_full[active_idx] = active_actions
            env.step(actions_full)
            obs_np = env.observe()
    finally:
        if was_training:
            agent.policy.train()

    return env.n_shots.copy()


def compare_agents(
    agent: "MLPAgent",
    num_games: int = 100,
    eval_seed: int = 10_000,
    greedy: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Compara MLP vs Bot Aleatorio vs Bot Heurístico Sencillo sobre los mismos mapas."""
    from battleship_rl.agents.baselines import (
        RandomBot,
        SimpleHeuristicBot,
        evaluate_baseline_bot,
    )

    fleets = generate_eval_fleets(num_games=num_games, seed=eval_seed)

    mlp_shots = evaluate_mlp_agent(agent, fleets, greedy=greedy)
    random_shots = evaluate_baseline_bot(RandomBot(seed=eval_seed + 1), fleets)
    heuristic_shots = evaluate_baseline_bot(
        SimpleHeuristicBot(seed=eval_seed + 2, randomize_hunt=True), fleets
    )

    return {
        "mlp": compute_block_stats(mlp_shots),
        "random": compute_block_stats(random_shots),
        "heuristic": compute_block_stats(heuristic_shots),
    }

"""Bucle de doble entrenamiento REINFORCE secuencial (por disparo + cada 20 rondas)."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from battleship_rl.agents.mlp_agent import MLPAgent
from battleship_rl.config import Config
from battleship_rl.game.constants import NUM_CELLS, OUTCOME_HIT, OUTCOME_MISS, OUTCOME_SUNK
from battleship_rl.game.engine_ref import BattleshipRefEnv
from battleship_rl.utils.checkpoint import (
    resolve_checkpoint_dir,
    save_checkpoint,
    save_metrics_csv,
)
from battleship_rl.utils.metrics import (
    compare_agents,
    compute_block_feedback,
    compute_block_stats,
)


class DualScaleReinforceTrainer:
    """Entrena un único MLP con rondas secuenciales (1..20), refuerzo por disparo y refuerzo por bloque.

    Usa una única tasa de aprendizaje (`config.learning_rate`) y pérdida directa:
        perdida = -(recompensa * log_prob)
    """

    def __init__(
        self,
        config: Config,
        agent: Optional[MLPAgent] = None,
        env: Optional[BattleshipRefEnv] = None,
    ) -> None:
        self.config = config
        self._set_seeds(config.seed)

        self.agent = agent if agent is not None else MLPAgent(config)
        self.env = env if env is not None else BattleshipRefEnv(seed=config.seed)

        self.optimizer = torch.optim.Adam(
            self.agent.policy.parameters(), lr=config.learning_rate
        )

        self.mu_ema: Optional[float] = None
        self.best_eval_mean: Optional[float] = None
        self.training_history: List[Dict[str, Any]] = []
        self.eval_history: List[Dict[str, Any]] = []

    @staticmethod
    def _set_seeds(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _immediate_rewards(self, outcomes: np.ndarray) -> np.ndarray:
        rewards = np.zeros(outcomes.shape, dtype=np.float32)
        rewards[(outcomes == OUTCOME_HIT) | (outcomes == OUTCOME_SUNK)] = (
            self.config.reward_hit
        )
        rewards[outcomes == OUTCOME_MISS] = self.config.reward_miss
        return rewards

    def _immediate_reward_scalar(self, outcome: int) -> float:
        if outcome == OUTCOME_HIT or outcome == OUTCOME_SUNK:
            return float(self.config.reward_hit)
        if outcome == OUTCOME_MISS:
            return float(self.config.reward_miss)
        return 0.0

    def train_block(self, block_idx: int) -> Dict[str, Any]:
        """Ejecuta `config.block_size` (20) rondas de forma secuencial y luego actualiza por bloque."""
        cfg = self.config
        device = self.agent.device
        self.agent.policy.train()

        buf_obs: List[np.ndarray] = []
        buf_masks: List[np.ndarray] = []
        buf_actions: List[int] = []
        buf_round_ids: List[int] = []

        shots_per_round = np.zeros((cfg.block_size,), dtype=np.int32)
        step_losses: List[float] = []
        step_entropies: List[float] = []
        total_hits = 0
        total_sunk = 0
        total_misses = 0

        # 1. Rondas estrictamente secuenciales: ronda 1 -> ronda 2 -> ... -> ronda 20
        for round_idx in range(cfg.block_size):
            obs_np = self.env.reset()

            for _ in range(NUM_CELLS):
                if self.env.done:
                    break

                valid_mask_np = self.env.valid_mask()
                actions_arr, log_probs, entropies = self.agent.select_actions(
                    obs_np[None, ...],
                    valid_mask_np[None, :],
                    greedy=False,
                    with_grad=True,
                )
                action = int(actions_arr[0])
                log_prob = log_probs[0]
                entropy = entropies[0]

                outcome, _, _ = self.env.step(action)
                if outcome == OUTCOME_HIT:
                    total_hits += 1
                elif outcome == OUTCOME_SUNK:
                    total_sunk += 1
                elif outcome == OUTCOME_MISS:
                    total_misses += 1

                # [Entrenamiento 1] Actualización inmediata tras ESTE disparo
                imm_reward = self._immediate_reward_scalar(outcome)
                step_loss = -(imm_reward * log_prob)
                if cfg.entropy_coef > 0.0:
                    step_loss = step_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.policy.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                step_losses.append(float(step_loss.detach().item()))
                step_entropies.append(float(entropy.detach().item()))

                # Guardar jugada para el refuerzo macro al terminar las 20 rondas
                buf_obs.append(obs_np.copy())
                buf_masks.append(valid_mask_np.copy())
                buf_actions.append(action)
                buf_round_ids.append(round_idx)

                obs_np = self.env.observe()

            shots_per_round[round_idx] = self.env.n_shots

        # 2. Estadísticas de las 20 rondas y señal macro del bloque
        stats = compute_block_stats(shots_per_round)
        block_signal, next_mu_ema, mean_term, var_factor = compute_block_feedback(
            mu_actual=stats["mean"],
            sigma_actual=stats["std"],
            mu_ema=self.mu_ema,
            config=cfg,
        )
        self.mu_ema = next_mu_ema

        # Ventaja por ronda dentro del bloque: señal global + mérito relativo opcional
        round_adv_np = block_signal + cfg.round_relative_weight * (
            stats["mean"] - shots_per_round.astype(np.float32)
        )
        round_adv_np = np.clip(
            round_adv_np, -cfg.clip_block_advantage, cfg.clip_block_advantage
        ).astype(np.float32)

        # 3. [Entrenamiento 2] Actualización macro sobre las 20 rondas (re-forward limpio)
        b_obs_t = torch.as_tensor(np.stack(buf_obs, axis=0), dtype=torch.float32, device=device)
        b_masks_t = torch.as_tensor(np.stack(buf_masks, axis=0), dtype=torch.bool, device=device)
        b_actions_t = torch.as_tensor(buf_actions, dtype=torch.int64, device=device)
        b_round_ids_t = torch.as_tensor(buf_round_ids, dtype=torch.int64, device=device)
        b_adv_t = torch.as_tensor(round_adv_np, dtype=torch.float32, device=device)[b_round_ids_t]

        block_losses: List[float] = []
        for _ in range(cfg.block_update_epochs):
            new_lp_t, entropies_t = self.agent.evaluate_actions(
                b_obs_t, b_masks_t, b_actions_t
            )
            block_loss = -(b_adv_t * new_lp_t).mean()
            if cfg.entropy_coef > 0.0:
                block_loss = block_loss - cfg.entropy_coef * entropies_t.mean()

            self.optimizer.zero_grad(set_to_none=True)
            block_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agent.policy.parameters(), cfg.max_grad_norm
            )
            self.optimizer.step()
            block_losses.append(float(block_loss.detach().item()))

        record: Dict[str, Any] = {
            "block": int(block_idx),
            "mean_shots": stats["mean"],
            "std_shots": stats["std"],
            "min_shots": stats["min"],
            "max_shots": stats["max"],
            "mu_ema": float(self.mu_ema),
            "block_signal": float(block_signal),
            "mean_term": float(mean_term),
            "var_factor": float(var_factor),
            "step_loss": float(np.mean(step_losses)) if step_losses else 0.0,
            "block_loss": float(np.mean(block_losses)) if block_losses else 0.0,
            "entropy": float(np.mean(step_entropies)) if step_entropies else 0.0,
            "hits": total_hits,
            "sunk": total_sunk,
            "misses": total_misses,
        }
        self.training_history.append(record)
        return record

    def train(
        self,
        num_blocks: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Ejecuta el entrenamiento completo a lo largo de `num_blocks` bloques."""
        cfg = self.config
        total_blocks = num_blocks if num_blocks is not None else cfg.num_blocks
        ckpt_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else resolve_checkpoint_dir(cfg.checkpoint_dir)
        )

        for block_idx in range(1, total_blocks + 1):
            metrics = self.train_block(block_idx)

            if verbose and (
                cfg.log_every_blocks > 0
                and (block_idx == 1 or block_idx % cfg.log_every_blocks == 0 or block_idx == total_blocks)
            ):
                rounds_done = block_idx * cfg.block_size
                print(
                    f"[Bloque {block_idx:4d}/{total_blocks} | Rondas {rounds_done:5d}] "
                    f"media={metrics['mean_shots']:.2f} std={metrics['std_shots']:.2f} "
                    f"min={metrics['min_shots']:.0f} max={metrics['max_shots']:.0f} "
                    f"ema={metrics['mu_ema']:.2f} señal={metrics['block_signal']:+.3f} "
                    f"ent={metrics['entropy']:.3f}"
                )

            if cfg.eval_every_blocks > 0 and (
                block_idx % cfg.eval_every_blocks == 0 or block_idx == total_blocks
            ):
                comp = compare_agents(
                    self.agent,
                    num_games=cfg.eval_num_games,
                    eval_seed=cfg.eval_seed_base,
                    greedy=cfg.eval_greedy,
                )
                eval_row = {
                    "block": int(block_idx),
                    "mlp_mean": comp["mlp"]["mean"],
                    "mlp_std": comp["mlp"]["std"],
                    "random_mean": comp["random"]["mean"],
                    "random_std": comp["random"]["std"],
                    "heuristic_mean": comp["heuristic"]["mean"],
                    "heuristic_std": comp["heuristic"]["std"],
                }
                self.eval_history.append(eval_row)

                if verbose:
                    print(
                        f"  ↳ Eval ({cfg.eval_num_games} mapas): "
                        f"MLP={eval_row['mlp_mean']:.2f}±{eval_row['mlp_std']:.2f} | "
                        f"Aleatorio={eval_row['random_mean']:.2f}±{eval_row['random_std']:.2f} | "
                        f"Heurístico={eval_row['heuristic_mean']:.2f}±{eval_row['heuristic_std']:.2f}"
                    )

                if self.best_eval_mean is None or eval_row["mlp_mean"] < self.best_eval_mean:
                    self.best_eval_mean = eval_row["mlp_mean"]
                    save_checkpoint(
                        ckpt_dir / cfg.best_checkpoint_filename,
                        self.agent.policy,
                        self.optimizer,
                        block_idx,
                        self.mu_ema,
                        self.best_eval_mean,
                        cfg,
                    )

            if cfg.save_every_blocks > 0 and (
                block_idx % cfg.save_every_blocks == 0 or block_idx == total_blocks
            ):
                save_checkpoint(
                    ckpt_dir / cfg.checkpoint_filename,
                    self.agent.policy,
                    self.optimizer,
                    block_idx,
                    self.mu_ema,
                    self.best_eval_mean,
                    cfg,
                )
                save_metrics_csv(self.training_history, ckpt_dir / cfg.metrics_filename)
                if self.eval_history:
                    save_metrics_csv(self.eval_history, ckpt_dir / cfg.eval_filename)

        return {
            "training_history": self.training_history,
            "eval_history": self.eval_history,
        }

"""Pruebas 4 y 7: Cálculo de bloque de 20 rondas y smoke test de gradientes PyTorch."""

import numpy as np
import torch

from battleship_rl.config import Config
from battleship_rl.learning.trainer import DualScaleReinforceTrainer
from battleship_rl.utils.metrics import compute_block_feedback, compute_block_stats


def test_04_block_stats_and_std_penalty() -> None:
    """Prueba 4: Media/std sobre 20 rondas y penalización estricta por mayor dispersión."""
    rounds = np.array(
        [42, 48, 55, 61, 39, 50, 46, 53, 58, 44, 49, 52, 47, 60, 41, 56, 51, 45, 54, 48],
        dtype=np.float32,
    )
    stats = compute_block_stats(rounds)
    assert stats["n_rounds"] == 20.0
    assert np.isclose(stats["mean"], float(np.mean(rounds)))
    assert np.isclose(stats["std"], float(np.std(rounds)))
    assert stats["min"] == 39.0
    assert stats["max"] == 61.0

    cfg = Config()
    sig_low_std, next_ema_1, _, _ = compute_block_feedback(
        mu_actual=50.0, sigma_actual=2.0, mu_ema=55.0, config=cfg
    )
    sig_high_std, next_ema_2, _, _ = compute_block_feedback(
        mu_actual=50.0, sigma_actual=12.0, mu_ema=55.0, config=cfg
    )

    # Si el promedio bajó, nunca puede volverse negativo, solo más pequeño en valor absoluto
    assert sig_low_std > sig_high_std > 0.0
    assert np.isclose(next_ema_1, next_ema_2)

    # Si el promedio subió, la varianza alta también reduce su magnitud absoluta sin cambiar signo
    neg_low_std, _, _, _ = compute_block_feedback(
        mu_actual=60.0, sigma_actual=2.0, mu_ema=55.0, config=cfg
    )
    neg_high_std, _, _, _ = compute_block_feedback(
        mu_actual=60.0, sigma_actual=12.0, mu_ema=55.0, config=cfg
    )
    assert abs(neg_low_std) > abs(neg_high_std)
    assert neg_low_std < neg_high_std < 0.0


def test_07_dual_update_pytorch_gradient_smoke_test() -> None:
    """Prueba 7: 2 bloques completos de 20 rondas ejecutan sin error de grafo ni NaN/Inf."""
    cfg = Config(
        device="cpu",
        input_dim=4,
        hidden_dims=(32, 16),
        output_dim=1,
        block_size=20,
        num_blocks=2,
        eval_every_blocks=0,
        save_every_blocks=0,
    )
    trainer = DualScaleReinforceTrainer(cfg)
    from battleship_rl.game.constants import OUTCOME_HIT, OUTCOME_MISS, OUTCOME_SUNK

    imm = trainer._immediate_rewards(np.array([OUTCOME_HIT, OUTCOME_SUNK, OUTCOME_MISS]))
    assert np.isclose(imm[0], cfg.reward_hit)
    assert np.isclose(imm[1], cfg.reward_hit)
    assert np.isclose(imm[2], cfg.reward_miss)

    initial_params = [p.detach().clone() for p in trainer.agent.policy.parameters()]

    m1 = trainer.train_block(1)
    m2 = trainer.train_block(2)

    assert 17.0 <= m1["mean_shots"] <= 100.0
    assert 17.0 <= m2["mean_shots"] <= 100.0
    assert np.isfinite(m1["step_loss"]) and np.isfinite(m1["block_loss"])
    assert np.isfinite(m2["step_loss"]) and np.isfinite(m2["block_loss"])

    changed = False
    for p0, p1 in zip(initial_params, trainer.agent.policy.parameters()):
        assert torch.all(torch.isfinite(p1))
        if not torch.allclose(p0, p1):
            changed = True
    assert changed, "Los pesos del MLP deben actualizarse tras el doble entrenamiento."

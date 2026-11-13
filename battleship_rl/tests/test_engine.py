"""Pruebas 1, 3 y 6: Integridad del tablero, sin disparos repetidos y masking en lotes vectorizados."""

import numpy as np
import pytest

from battleship_rl.game.constants import (
    NUM_CELLS,
    OUTCOME_NOOP,
    SHIP_LENGTHS,
    SUNK,
    TOTAL_SHIP_CELLS,
    UNSHOT,
)
from battleship_rl.game.engine_ref import BattleshipRefEnv, generate_random_fleet, validate_ship_id_grid
from battleship_rl.game.engine_vec import BattleshipVecEnv


def test_01_fleet_integrity_many_random_boards() -> None:
    """Prueba 1: 200 mapas aleatorios tienen 17 casillas exactas, 5 barcos sin solapamientos."""
    rng = np.random.default_rng(123)
    for _ in range(200):
        ship_id = generate_random_fleet(rng)
        validate_ship_id_grid(ship_id)
        assert int(np.sum(ship_id >= 0)) == TOTAL_SHIP_CELLS
        for sid, expected_len in enumerate(SHIP_LENGTHS):
            assert int(np.sum(ship_id == sid)) == expected_len


def test_03_no_repeated_shots_and_bounded_game_length() -> None:
    """Prueba 3: La máscara impide disparos repetidos y toda partida termina en <= 100 tiros."""
    env = BattleshipRefEnv(seed=7)
    rng = np.random.default_rng(7)
    seen = set()

    while not env.done:
        mask = env.valid_mask()
        valid_cells = np.flatnonzero(mask)
        assert valid_cells.size == NUM_CELLS - env.n_shots
        for c in seen:
            assert not mask[c]

        action = int(rng.choice(valid_cells))
        seen.add(action)
        env.step(action)

        # Intentar repetir disparo sobre una casilla ya atacada debe fallar explícitamente
        if not env.done:
            with pytest.raises(ValueError, match="ya fue atacada"):
                env.step(action)

    assert 17 <= env.n_shots <= 100
    assert int(np.sum(env.shots == SUNK)) == TOTAL_SHIP_CELLS


def test_06_vectorized_async_done_masking() -> None:
    """Prueba 6: En un lote de 20 partidas, las terminadas congelan n_shots y shots."""
    env = BattleshipVecEnv(batch_size=20, seed=99)
    rng = np.random.default_rng(99)

    finished_snapshots = {}

    for _ in range(NUM_CELLS):
        if np.all(env.done):
            break

        valid_mask = env.valid_mask()
        actions = np.zeros((20,), dtype=np.int64)
        for b in range(20):
            if env.done[b]:
                # Aunque pasemos una acción cualquiera para env ya terminado, debe ser NOOP
                actions[b] = 0
            else:
                choices = np.flatnonzero(valid_mask[b])
                actions[b] = int(rng.choice(choices))

        outcomes, dones, _ = env.step(actions)

        for b in range(20):
            if b in finished_snapshots:
                prev_n_shots, prev_shots = finished_snapshots[b]
                assert outcomes[b] == OUTCOME_NOOP
                assert env.n_shots[b] == prev_n_shots
                assert np.array_equal(env.shots[b], prev_shots)
            elif dones[b]:
                finished_snapshots[b] = (int(env.n_shots[b]), env.shots[b].copy())

    assert np.all(env.done)
    assert len(finished_snapshots) == 20
    assert np.all((env.n_shots >= 17) & (env.n_shots <= 100))
    assert np.all(env.shots != UNSHOT, axis=1).sum() <= 20

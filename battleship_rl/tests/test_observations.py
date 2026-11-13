"""Prueba 2: Consistencia exacta de la matriz (100, 4) para el MLP de 4 neuronas."""

import numpy as np

from battleship_rl.game.constants import BOARD_SIZE, MAX_SHIP_LENGTH, NUM_CELLS
from battleship_rl.game.engine_ref import BattleshipRefEnv
from battleship_rl.game.engine_vec import BattleshipVecEnv
from battleship_rl.utils.heuristic_map import get_heuristic_map_flat


def _deterministic_fleet() -> np.ndarray:
    """Flota fija conocida para probar el descenso progresivo del barco vivo más pequeño."""
    grid = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int8)
    grid[0, 0:5] = 0  # Portaaviones (5)
    grid[2, 0:4] = 1  # Acorazado (4)
    grid[4, 0:3] = 2  # Crucero (3)
    grid[6, 0:3] = 3  # Submarino (3)
    grid[8, 0:2] = 4  # Destructor (2)
    return grid.reshape(NUM_CELLS)


def test_02_observation_matrix_4_features_and_min_alive_ship() -> None:
    """Prueba 2: Matriz (100, 4), canales [0, 1], mapa heurístico constante y min_alive_len."""
    fleet = _deterministic_fleet()
    env_ref = BattleshipRefEnv()
    env_vec = BattleshipVecEnv(batch_size=1)

    obs_ref = env_ref.reset(fixed_ship_id=fleet)
    obs_vec = env_vec.reset(fixed_ship_ids=fleet[None, :])
    assert obs_ref.shape == (NUM_CELLS, 4)
    assert obs_vec.shape == (1, NUM_CELLS, 4)
    assert np.allclose(obs_ref, obs_vec[0])

    static_heur = get_heuristic_map_flat()
    assert np.all((static_heur > 0.0) & (static_heur <= 1.0))

    # Canal 0 inicia en 0 (sin HIT activos); Canal 2 es el mapa heurístico; Canal 3 inicia en 2/5 = 0.4
    assert np.all(obs_ref[:, 0] == 0.0)
    assert np.allclose(obs_ref[:, 2], static_heur)
    assert np.allclose(obs_ref[:, 3], 2.0 / MAX_SHIP_LENGTH)
    assert np.all((obs_ref >= 0.0) & (obs_ref <= 1.0))

    # Disparo al agua en (9, 9) -> índice 99
    env_ref.step(99)
    env_vec.step(np.array([99]))
    obs_ref = env_ref.observe()
    obs_vec = env_vec.observe()
    assert np.allclose(obs_ref, obs_vec[0])
    assert obs_ref[99, 1] == 0.0  # ya atacada -> canal 1 en 0.0

    # Primer HIT sobre Portaaviones en (0, 0) -> índice 0: vecinos (0, 1)=1 y (1, 0)=10 pasan a 0.5
    env_ref.step(0)
    env_vec.step(np.array([0]))
    obs_hit1 = env_ref.observe()
    assert obs_hit1[1, 0] == 0.5
    assert obs_hit1[10, 0] == 0.5

    # Segundo HIT en (0, 1) -> índice 1: extensión alineada (0, 2)=2 pasa a 1.0
    env_ref.step(1)
    env_vec.step(np.array([1]))
    obs_hit2 = env_ref.observe()
    assert obs_hit2[2, 0] == 1.0

    # Terminar de hundir Portaaviones (0, 2..4) -> al pasar a SUNK el canal 0 vuelve a 0.0
    for idx in (2, 3, 4):
        env_ref.step(idx)
        env_vec.step(np.array([idx]))
    obs_sunk = env_ref.observe()
    assert np.all(obs_sunk[:, 0] == 0.0)
    assert np.allclose(obs_sunk[:, 2], static_heur)

    # Hundir Destructor (80, 81) -> min_alive sube a 3/5 = 0.6
    for idx in (80, 81):
        env_ref.step(idx)
        env_vec.step(np.array([idx]))
    assert np.allclose(env_ref.observe()[:, 3], 3.0 / MAX_SHIP_LENGTH)

    # Hundir Crucero (40..42) -> aún queda Submarino (3) -> sigue en 0.6
    for idx in (40, 41, 42):
        env_ref.step(idx)
        env_vec.step(np.array([idx]))
    assert np.allclose(env_ref.observe()[:, 3], 3.0 / MAX_SHIP_LENGTH)

    # Hundir Submarino (60..62) -> min_alive sube a 4/5 = 0.8
    for idx in (60, 61, 62):
        env_ref.step(idx)
        env_vec.step(np.array([idx]))
    assert np.allclose(env_ref.observe()[:, 3], 4.0 / MAX_SHIP_LENGTH)

    # Hundir Acorazado (20..23) -> fin de partida -> min_alive = 0.0
    for idx in (20, 21, 22, 23):
        env_ref.step(idx)
        env_vec.step(np.array([idx]))
    obs_end = env_ref.observe()
    assert env_ref.done and env_vec.done[0]
    assert np.allclose(obs_end, env_vec.observe()[0])
    assert np.all(obs_end[:, 3] == 0.0)
    assert np.allclose(obs_end[:, 2], static_heur)

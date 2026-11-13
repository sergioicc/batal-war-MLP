"""Cálculo del mapa heurístico estático inicial de densidad de probabilidad."""

from functools import lru_cache
from typing import Tuple

import numpy as np

from battleship_rl.game.constants import BOARD_SIZE, NUM_CELLS, SHIP_LENGTHS


def compute_initial_heuristic_map(
    board_size: int = BOARD_SIZE,
    ship_lengths: Tuple[int, ...] = SHIP_LENGTHS,
    normalize: bool = True,
) -> np.ndarray:
    """Calcula la densidad de probabilidad inicial sobre un tablero vacío.

    Recorre todas las colocaciones horizontales y verticales válidas de cada
    barco de la flota reglamentaria y acumula cuántas configuraciones cubren
    cada casilla.

    Args:
        board_size: Tamaño del tablero cuadrado (por defecto 10).
        ship_lengths: Tupla con longitudes de los barcos.
        normalize: Si es True, divide por el valor máximo para escalar a [0, 1].

    Returns:
        Matriz (board_size, board_size) en float32.
    """
    counts = np.zeros((board_size, board_size), dtype=np.float32)

    for length in ship_lengths:
        # Colocaciones horizontales válidas
        for r in range(board_size):
            for c in range(board_size - length + 1):
                counts[r, c : c + length] += 1.0

        # Colocaciones verticales válidas
        for r in range(board_size - length + 1):
            for c in range(board_size):
                counts[r : r + length, c] += 1.0

    if normalize:
        max_val = float(counts.max())
        if max_val > 0.0:
            counts /= max_val

    return counts


@lru_cache(maxsize=1)
def get_heuristic_map_flat() -> np.ndarray:
    """Devuelve el mapa heurístico inicial normalizado [0, 1] aplanado (100,)."""
    grid = compute_initial_heuristic_map(BOARD_SIZE, SHIP_LENGTHS, normalize=True)
    flat = grid.reshape(NUM_CELLS).astype(np.float32)
    flat.setflags(write=False)
    return flat

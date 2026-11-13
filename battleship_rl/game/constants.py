"""Constantes y reglas oficiales del tablero y flota de Batalla Naval."""

from typing import Tuple

BOARD_SIZE: int = 10
NUM_CELLS: int = BOARD_SIZE * BOARD_SIZE  # 100

# Flota reglamentaria: (nombre, longitud). Total = 17 casillas.
# Nota: Crucero y Submarino tienen longitud 3 pero IDs distintos (2 y 3).
SHIPS: Tuple[Tuple[str, int], ...] = (
    ("Portaaviones", 5),
    ("Acorazado", 4),
    ("Crucero", 3),
    ("Submarino", 3),
    ("Destructor", 2),
)

SHIP_NAMES: Tuple[str, ...] = tuple(name for name, _ in SHIPS)
SHIP_LENGTHS: Tuple[int, ...] = tuple(length for _, length in SHIPS)
NUM_SHIPS: int = len(SHIPS)
TOTAL_SHIP_CELLS: int = sum(SHIP_LENGTHS)  # 17
MAX_SHIP_LENGTH: int = max(SHIP_LENGTHS)  # 5

# Estados visibles en el tablero de disparos (`shots`)
UNSHOT: int = 0
MISS: int = 1
HIT: int = 2
SUNK: int = 3

# Códigos de resultado por paso (`step`)
OUTCOME_NOOP: int = -1  # Partida ya terminada en lote vectorizado
OUTCOME_MISS: int = 0
OUTCOME_HIT: int = 1
OUTCOME_SUNK: int = 2

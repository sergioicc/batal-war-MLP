"""Motores de juego y reglas de Batalla Naval."""

from battleship_rl.game.constants import (
    BOARD_SIZE,
    HIT,
    MAX_SHIP_LENGTH,
    MISS,
    NUM_CELLS,
    NUM_SHIPS,
    OUTCOME_HIT,
    OUTCOME_MISS,
    OUTCOME_NOOP,
    OUTCOME_SUNK,
    SHIP_LENGTHS,
    SHIP_NAMES,
    SHIPS,
    SUNK,
    TOTAL_SHIP_CELLS,
    UNSHOT,
)
from battleship_rl.game.engine_ref import BattleshipRefEnv, generate_random_fleet, validate_ship_id_grid
from battleship_rl.game.engine_vec import BattleshipVecEnv

__all__ = [
    "BOARD_SIZE",
    "NUM_CELLS",
    "SHIPS",
    "SHIP_NAMES",
    "SHIP_LENGTHS",
    "NUM_SHIPS",
    "TOTAL_SHIP_CELLS",
    "MAX_SHIP_LENGTH",
    "UNSHOT",
    "MISS",
    "HIT",
    "SUNK",
    "OUTCOME_NOOP",
    "OUTCOME_MISS",
    "OUTCOME_HIT",
    "OUTCOME_SUNK",
    "BattleshipRefEnv",
    "BattleshipVecEnv",
    "generate_random_fleet",
    "validate_ship_id_grid",
]

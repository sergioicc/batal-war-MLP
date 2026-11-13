"""Motor de referencia orientado a objetos para validación de reglas de Batalla Naval."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from battleship_rl.game.constants import (
    BOARD_SIZE,
    HIT,
    MAX_SHIP_LENGTH,
    MISS,
    NUM_CELLS,
    NUM_SHIPS,
    OUTCOME_HIT,
    OUTCOME_MISS,
    OUTCOME_SUNK,
    SHIP_LENGTHS,
    SHIP_NAMES,
    SUNK,
    TOTAL_SHIP_CELLS,
    UNSHOT,
)
from battleship_rl.utils.heuristic_map import get_heuristic_map_flat


@dataclass
class ShipState:
    """Representa el estado de un barco individual en la partida."""

    ship_id: int
    name: str
    length: int
    cells: Tuple[int, ...]
    hits: int = 0

    @property
    def is_sunk(self) -> bool:
        return self.hits >= self.length


def generate_random_fleet(rng: np.random.Generator) -> np.ndarray:
    """Genera una colocación aleatoria válida de los 5 barcos en un tablero 10x10.

    Returns:
        Arreglo int8 de forma (100,) con -1 para agua y 0..4 para cada barco.
    """
    while True:
        grid = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int8)
        success = True
        # Barajar el orden de los barcos para evitar cualquier sesgo posicional
        ship_order = rng.permutation(NUM_SHIPS)

        for sid_np in ship_order:
            sid = int(sid_np)
            length = SHIP_LENGTHS[sid]
            candidates: List[Tuple[int, int, int]] = []  # (r, c, orientation: 0=H, 1=V)

            # Candidatos horizontales
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE - length + 1):
                    if np.all(grid[r, c : c + length] == -1):
                        candidates.append((r, c, 0))

            # Candidatos verticales
            for r in range(BOARD_SIZE - length + 1):
                for c in range(BOARD_SIZE):
                    if np.all(grid[r : r + length, c] == -1):
                        candidates.append((r, c, 1))

            if not candidates:
                success = False
                break

            idx = int(rng.integers(0, len(candidates)))
            r, c, orient = candidates[idx]
            if orient == 0:
                grid[r, c : c + length] = sid
            else:
                grid[r : r + length, c] = sid

        if success:
            return grid.reshape(NUM_CELLS)


def validate_ship_id_grid(ship_id: np.ndarray) -> None:
    """Verifica que un tablero de flota cumpla estrictamente las reglas oficiales."""
    arr = np.asarray(ship_id)
    if arr.shape == (NUM_CELLS,):
        grid = arr.reshape(BOARD_SIZE, BOARD_SIZE)
    elif arr.shape == (BOARD_SIZE, BOARD_SIZE):
        grid = arr
    else:
        raise ValueError(f"Forma inválida para ship_id: {arr.shape}")

    occupied = int(np.sum(grid >= 0))
    if occupied != TOTAL_SHIP_CELLS:
        raise ValueError(
            f"La flota debe ocupar exactamente {TOTAL_SHIP_CELLS} casillas; se encontraron {occupied}."
        )

    for sid, expected_len in enumerate(SHIP_LENGTHS):
        coords = np.argwhere(grid == sid)
        if len(coords) != expected_len:
            raise ValueError(
                f"Barco {sid} ({SHIP_NAMES[sid]}) debe tener longitud {expected_len}, tiene {len(coords)}."
            )
        rows = coords[:, 0]
        cols = coords[:, 1]
        is_horizontal = np.all(rows == rows[0]) and np.array_equal(
            np.sort(cols), np.arange(cols.min(), cols.min() + expected_len)
        )
        is_vertical = np.all(cols == cols[0]) and np.array_equal(
            np.sort(rows), np.arange(rows.min(), rows.min() + expected_len)
        )
        if not (is_horizontal or is_vertical):
            raise ValueError(
                f"Barco {sid} ({SHIP_NAMES[sid]}) no forma una línea recta continua."
            )


def build_cell_features_batch(
    shots_batch: np.ndarray,
    min_alive_lengths: np.ndarray,
    heuristic_map_flat: np.ndarray,
) -> np.ndarray:
    """Construye la matriz de observación (B, 100, 4) para el MLP de 4 neuronas de entrada.

    Canales por casilla (r, c):
      0 - Contexto de ataques previos (`HIT` activos): 1.0 si extiende línea de 2+ HIT,
          0.5 si tiene al menos un HIT ortogonal adyacente, 0.0 en otro caso.
      1 - Contexto de casillas por atacar (`UNSHOT`): proporción de vecinos ortogonales
          aún no atacados (0..1), y 0.0 si la casilla misma ya fue atacada.
      2 - Valor del mapa heurístico inicial normalizado [0, 1] en la casilla.
      3 - Longitud normalizada del barco vivo más pequeño (`min_alive_len / 5.0`).
    """
    B = shots_batch.shape[0]
    shots_2d = shots_batch.reshape(B, BOARD_SIZE, BOARD_SIZE)

    is_unshot = shots_2d == UNSHOT
    is_hit = shots_2d == HIT

    # Vecinos ortogonales con HIT activo
    hit_up = np.zeros_like(is_hit)
    hit_down = np.zeros_like(is_hit)
    hit_left = np.zeros_like(is_hit)
    hit_right = np.zeros_like(is_hit)
    hit_up[:, 1:, :] = is_hit[:, :-1, :]
    hit_down[:, :-1, :] = is_hit[:, 1:, :]
    hit_left[:, :, 1:] = is_hit[:, :, :-1]
    hit_right[:, :, :-1] = is_hit[:, :, 1:]

    has_ortho_hit = hit_up | hit_down | hit_left | hit_right

    # Extensión alineada de 2+ HIT consecutivos (p. ej. (r, c-1) y (r, c-2) son HIT)
    hit_up2 = np.zeros_like(is_hit)
    hit_down2 = np.zeros_like(is_hit)
    hit_left2 = np.zeros_like(is_hit)
    hit_right2 = np.zeros_like(is_hit)
    hit_up2[:, 2:, :] = is_hit[:, :-2, :]
    hit_down2[:, :-2, :] = is_hit[:, 2:, :]
    hit_left2[:, :, 2:] = is_hit[:, :, :-2]
    hit_right2[:, :, :-2] = is_hit[:, :, 2:]

    aligned_hit = (
        (hit_up & hit_up2)
        | (hit_down & hit_down2)
        | (hit_left & hit_left2)
        | (hit_right & hit_right2)
        | (hit_up & hit_down)
        | (hit_left & hit_right)
    )

    ch0 = np.where(aligned_hit, 1.0, np.where(has_ortho_hit, 0.5, 0.0)).astype(np.float32)
    ch0 = np.where(is_unshot, ch0, 0.0).reshape(B, NUM_CELLS)

    # Vecinos ortogonales aún sin atacar (UNSHOT)
    un_up = np.zeros_like(is_unshot)
    un_down = np.zeros_like(is_unshot)
    un_left = np.zeros_like(is_unshot)
    un_right = np.zeros_like(is_unshot)
    un_up[:, 1:, :] = is_unshot[:, :-1, :]
    un_down[:, :-1, :] = is_unshot[:, 1:, :]
    un_left[:, :, 1:] = is_unshot[:, :, :-1]
    un_right[:, :, :-1] = is_unshot[:, :, 1:]

    ortho_unshot_frac = (
        un_up.astype(np.float32)
        + un_down.astype(np.float32)
        + un_left.astype(np.float32)
        + un_right.astype(np.float32)
    ) / 4.0
    ch1 = np.where(is_unshot, ortho_unshot_frac, 0.0).reshape(B, NUM_CELLS)

    obs = np.empty((B, NUM_CELLS, 4), dtype=np.float32)
    obs[:, :, 0] = ch0
    obs[:, :, 1] = ch1
    obs[:, :, 2] = heuristic_map_flat[None, :]
    obs[:, :, 3] = (min_alive_lengths.astype(np.float32) / float(MAX_SHIP_LENGTH))[:, None]
    return obs


class BattleshipRefEnv:
    """Motor de referencia OOP para una partida en modo solitario (tablero atacante)."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.heuristic_map = get_heuristic_map_flat()
        self.ship_id: np.ndarray = np.full(NUM_CELLS, -1, dtype=np.int8)
        self.shots: np.ndarray = np.full(NUM_CELLS, UNSHOT, dtype=np.int8)
        self.ships: Dict[int, ShipState] = {}
        self.n_shots: int = 0
        self.done: bool = False
        self.reset()

    def reset(self, fixed_ship_id: Optional[np.ndarray] = None) -> np.ndarray:
        """Reinicia la partida con una flota aleatoria o una predefinida."""
        if fixed_ship_id is not None:
            validate_ship_id_grid(fixed_ship_id)
            self.ship_id = np.asarray(fixed_ship_id, dtype=np.int8).reshape(NUM_CELLS).copy()
        else:
            self.ship_id = generate_random_fleet(self.rng)

        self.shots = np.full(NUM_CELLS, UNSHOT, dtype=np.int8)
        self.ships = {}
        for sid in range(NUM_SHIPS):
            cells = tuple(int(idx) for idx in np.flatnonzero(self.ship_id == sid))
            self.ships[sid] = ShipState(
                ship_id=sid,
                name=SHIP_NAMES[sid],
                length=SHIP_LENGTHS[sid],
                cells=cells,
                hits=0,
            )
        self.n_shots = 0
        self.done = False
        return self.observe()

    def valid_mask(self) -> np.ndarray:
        """Máscara booleana (100,) con True en casillas aún no atacadas."""
        return self.shots == UNSHOT

    def min_alive_ship_length(self) -> int:
        """Longitud del barco vivo más pequeño (0 si toda la flota está hundida)."""
        alive_lengths = [s.length for s in self.ships.values() if not s.is_sunk]
        return min(alive_lengths) if alive_lengths else 0

    def observe(self) -> np.ndarray:
        """Construye la matriz de observación (100, 4) para el MLP de 4 neuronas."""
        min_len = np.array([self.min_alive_ship_length()], dtype=np.int32)
        return build_cell_features_batch(
            self.shots[None, :], min_len, self.heuristic_map
        )[0]

    def step(self, action: int) -> Tuple[int, bool, Dict[str, int]]:
        """Ejecuta un disparo en la casilla `action` (0..99)."""
        if self.done:
            raise RuntimeError("La partida ya finalizó; llama a reset() antes de continuar.")
        if not (0 <= action < NUM_CELLS):
            raise ValueError(f"Acción fuera de rango: {action}")
        if self.shots[action] != UNSHOT:
            raise ValueError(f"Casilla {action} ya fue atacada previamente.")

        self.n_shots += 1
        sid = int(self.ship_id[action])
        info: Dict[str, int] = {"ship_id": sid, "sunk_ship_length": 0}

        if sid == -1:
            self.shots[action] = MISS
            outcome = OUTCOME_MISS
        else:
            ship = self.ships[sid]
            ship.hits += 1
            if ship.is_sunk:
                for c in ship.cells:
                    self.shots[c] = SUNK
                outcome = OUTCOME_SUNK
                info["sunk_ship_length"] = ship.length
                if all(s.is_sunk for s in self.ships.values()):
                    self.done = True
            else:
                self.shots[action] = HIT
                outcome = OUTCOME_HIT

        return outcome, self.done, info

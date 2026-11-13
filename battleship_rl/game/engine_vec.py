"""Motor vectorizado por lotes para Batalla Naval en modo atacante solitario."""

from typing import Dict, Optional, Tuple

import numpy as np

from battleship_rl.game.constants import (
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
    SUNK,
    UNSHOT,
)
from battleship_rl.game.engine_ref import (
    build_cell_features_batch,
    generate_random_fleet,
    validate_ship_id_grid,
)
from battleship_rl.utils.heuristic_map import get_heuristic_map_flat


class BattleshipVecEnv:
    """Motor vectorizado que simula `batch_size` partidas independientes en paralelo.

    Maneja correctamente lotes donde las partidas terminan en distinto número de turnos:
    cuando `done[b]` es True, las llamadas posteriores a `step()` congelan el estado
    y el contador `n_shots[b]` de esa partida, devolviendo `OUTCOME_NOOP`.
    """

    def __init__(self, batch_size: int = 20, seed: Optional[int] = None) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size debe ser >= 1")
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.heuristic_map = get_heuristic_map_flat()
        self._ship_lengths_arr = np.asarray(SHIP_LENGTHS, dtype=np.int32)

        self.ship_id = np.full((batch_size, NUM_CELLS), -1, dtype=np.int8)
        self.shots = np.full((batch_size, NUM_CELLS), UNSHOT, dtype=np.int8)
        self.ship_hits = np.zeros((batch_size, NUM_SHIPS), dtype=np.int8)
        self.alive_ships = np.ones((batch_size, NUM_SHIPS), dtype=bool)
        self.n_shots = np.zeros((batch_size,), dtype=np.int32)
        self.done = np.zeros((batch_size,), dtype=bool)
        self.reset()

    def reset(
        self,
        fixed_ship_ids: Optional[np.ndarray] = None,
        validate_fixed: bool = False,
    ) -> np.ndarray:
        """Reinicia todas las partidas del lote."""
        if fixed_ship_ids is not None:
            arr = np.asarray(fixed_ship_ids, dtype=np.int8)
            if arr.shape != (self.batch_size, NUM_CELLS):
                raise ValueError(
                    f"fixed_ship_ids debe tener forma ({self.batch_size}, {NUM_CELLS}), recibido {arr.shape}"
                )
            if validate_fixed:
                for b in range(self.batch_size):
                    validate_ship_id_grid(arr[b])
            self.ship_id = arr.copy()
        else:
            for b in range(self.batch_size):
                self.ship_id[b] = generate_random_fleet(self.rng)

        self.shots.fill(UNSHOT)
        self.ship_hits.fill(0)
        self.alive_ships.fill(True)
        self.n_shots.fill(0)
        self.done.fill(False)
        return self.observe()

    def valid_mask(self) -> np.ndarray:
        """Máscara booleana (B, 100) con True en casillas aún no atacadas."""
        return self.shots == UNSHOT

    def min_alive_ship_lengths(self) -> np.ndarray:
        """Vector (B,) con la longitud del barco vivo más pequeño (0 si ya terminó)."""
        # Reemplaza barcos hundidos por un valor sentinela mayor que cualquier longitud
        sentinel = MAX_SHIP_LENGTH + 1
        lengths_matrix = np.where(
            self.alive_ships,
            self._ship_lengths_arr[None, :],
            sentinel,
        )
        mins = lengths_matrix.min(axis=1)
        mins = np.where(self.done, 0, mins)
        return mins.astype(np.int32)

    def observe(self) -> np.ndarray:
        """Construye el tensor de observación de forma (B, 100, 4) en float32."""
        return build_cell_features_batch(
            self.shots,
            self.min_alive_ship_lengths(),
            self.heuristic_map,
        )

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Aplica un vector de disparos (B,) sobre las partidas aún activas."""
        actions_arr = np.asarray(actions, dtype=np.int64)
        if actions_arr.shape != (self.batch_size,):
            raise ValueError(
                f"actions debe tener forma ({self.batch_size},), recibido {actions_arr.shape}"
            )

        outcomes = np.full((self.batch_size,), OUTCOME_NOOP, dtype=np.int32)
        sunk_lengths = np.zeros((self.batch_size,), dtype=np.int32)

        active_indices = np.flatnonzero(~self.done)
        if active_indices.size > 0:
            active_actions = actions_arr[active_indices]
            if np.any((active_actions < 0) | (active_actions >= NUM_CELLS)):
                raise ValueError("Alguna acción está fuera del rango 0..99.")

            already_shot = self.shots[active_indices, active_actions] != UNSHOT
            if np.any(already_shot):
                bad_env = int(active_indices[np.flatnonzero(already_shot)[0]])
                bad_act = int(actions_arr[bad_env])
                raise ValueError(
                    f"Disparo repetido en entorno {bad_env}, casilla {bad_act}."
                )

            self.n_shots[active_indices] += 1

            for b in active_indices:
                a = int(actions_arr[b])
                sid = int(self.ship_id[b, a])
                if sid == -1:
                    self.shots[b, a] = MISS
                    outcomes[b] = OUTCOME_MISS
                else:
                    self.ship_hits[b, sid] += 1
                    ship_len = SHIP_LENGTHS[sid]
                    if int(self.ship_hits[b, sid]) >= ship_len:
                        self.alive_ships[b, sid] = False
                        self.shots[b, self.ship_id[b] == sid] = SUNK
                        outcomes[b] = OUTCOME_SUNK
                        sunk_lengths[b] = ship_len
                        if not np.any(self.alive_ships[b]):
                            self.done[b] = True
                    else:
                        self.shots[b, a] = HIT
                        outcomes[b] = OUTCOME_HIT

        info = {
            "n_shots": self.n_shots.copy(),
            "sunk_lengths": sunk_lengths,
            "all_done": np.array(bool(np.all(self.done))),
        }
        return outcomes, self.done.copy(), info

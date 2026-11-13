"""Bots de referencia para comparación: Bot Aleatorio y Bot Heurístico Sencillo (Hunt & Target)."""

from typing import List, Optional, Protocol

import numpy as np

from battleship_rl.game.constants import BOARD_SIZE, HIT, NUM_CELLS, UNSHOT
from battleship_rl.game.engine_vec import BattleshipVecEnv


class BaselineBot(Protocol):
    name: str

    def select_action(self, shots_1d: np.ndarray) -> int:
        ...


class RandomBot:
    """Bot Aleatorio (límite inferior): elige uniformemente al azar entre casillas no atacadas."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.name = "Bot Aleatorio"
        self.rng = np.random.default_rng(seed)

    def select_action(self, shots_1d: np.ndarray) -> int:
        unshot = np.flatnonzero(shots_1d == UNSHOT)
        if unshot.size == 0:
            raise ValueError("No quedan casillas válidas por atacar.")
        idx = int(self.rng.integers(0, unshot.size))
        return int(unshot[idx])


class SimpleHeuristicBot:
    """Bot Heurístico Sencillo con modo Hunt (cada 2 espacios) y Target (caza ortogonal).

    - En modo **Hunt** (sin casillas `HIT` activas): recorre casillas con paridad
      de tablero de ajedrez `(r + c) % 2 == 0` cada dos espacios; si se agotan,
      cubre las casillas restantes `(r + c) % 2 == 1`.
    - En modo **Target** (cuando hay al menos una casilla `HIT` activa de un barco
      aún no hundido): prioriza extender ejes alineados de `HIT` adyacentes y,
      en su defecto, ataca los vecinos ortogonales disponibles alrededor de los `HIT`.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        randomize_hunt: bool = True,
    ) -> None:
        self.name = "Bot Heurístico Sencillo (Hunt & Target)"
        self.rng = np.random.default_rng(seed)
        self.randomize_hunt = randomize_hunt

        # Precalcular índices de paridad par (tablero de ajedrez) e impar
        self._even_cells = np.array(
            [
                r * BOARD_SIZE + c
                for r in range(BOARD_SIZE)
                for c in range(BOARD_SIZE)
                if (r + c) % 2 == 0
            ],
            dtype=np.int64,
        )
        self._odd_cells = np.array(
            [
                r * BOARD_SIZE + c
                for r in range(BOARD_SIZE)
                for c in range(BOARD_SIZE)
                if (r + c) % 2 == 1
            ],
            dtype=np.int64,
        )

    @staticmethod
    def _orthogonal_neighbors(cell: int) -> List[int]:
        r, c = divmod(cell, BOARD_SIZE)
        neighbors: List[int] = []
        if r > 0:
            neighbors.append((r - 1) * BOARD_SIZE + c)
        if r < BOARD_SIZE - 1:
            neighbors.append((r + 1) * BOARD_SIZE + c)
        if c > 0:
            neighbors.append(r * BOARD_SIZE + (c - 1))
        if c < BOARD_SIZE - 1:
            neighbors.append(r * BOARD_SIZE + (c + 1))
        return neighbors

    def _find_target_candidates(self, shots_1d: np.ndarray) -> List[int]:
        active_hits = np.flatnonzero(shots_1d == HIT)
        if active_hits.size == 0:
            return []

        hit_set = set(int(h) for h in active_hits)
        aligned_candidates: List[int] = []
        seen_aligned = set()

        # Si hay 2+ hits adyacentes en misma fila o columna, priorizar los extremos del eje
        if active_hits.size >= 2:
            for h in active_hits:
                r, c = divmod(int(h), BOARD_SIZE)
                # Eje horizontal
                has_h_neighbor = ((r * BOARD_SIZE + (c - 1)) in hit_set if c > 0 else False) or (
                    (r * BOARD_SIZE + (c + 1)) in hit_set if c < BOARD_SIZE - 1 else False
                )
                if has_h_neighbor:
                    for nc in (c - 1, c + 1):
                        if 0 <= nc < BOARD_SIZE:
                            cand = r * BOARD_SIZE + nc
                            if shots_1d[cand] == UNSHOT and cand not in seen_aligned:
                                aligned_candidates.append(cand)
                                seen_aligned.add(cand)

                # Eje vertical
                has_v_neighbor = (((r - 1) * BOARD_SIZE + c) in hit_set if r > 0 else False) or (
                    ((r + 1) * BOARD_SIZE + c) in hit_set if r < BOARD_SIZE - 1 else False
                )
                if has_v_neighbor:
                    for nr in (r - 1, r + 1):
                        if 0 <= nr < BOARD_SIZE:
                            cand = nr * BOARD_SIZE + c
                            if shots_1d[cand] == UNSHOT and cand not in seen_aligned:
                                aligned_candidates.append(cand)
                                seen_aligned.add(cand)

        if aligned_candidates:
            return aligned_candidates

        # En caso contrario, todos los vecinos ortogonales sin disparar de los HIT activos
        general_candidates: List[int] = []
        seen = set()
        for h in active_hits:
            for nb in self._orthogonal_neighbors(int(h)):
                if shots_1d[nb] == UNSHOT and nb not in seen:
                    general_candidates.append(nb)
                    seen.add(nb)
        return general_candidates

    def select_action(self, shots_1d: np.ndarray) -> int:
        # 1. Modo Target si existen HIT activos
        target_candidates = self._find_target_candidates(shots_1d)
        if target_candidates:
            if self.randomize_hunt and len(target_candidates) > 1:
                idx = int(self.rng.integers(0, len(target_candidates)))
                return target_candidates[idx]
            return target_candidates[0]

        # 2. Modo Hunt: barrido cada dos espacios (paridad par primero)
        even_valid = self._even_cells[shots_1d[self._even_cells] == UNSHOT]
        if even_valid.size > 0:
            if self.randomize_hunt:
                idx = int(self.rng.integers(0, even_valid.size))
                return int(even_valid[idx])
            return int(even_valid[0])

        # 3. Completar casillas impares restantes si ya se cubrió la paridad par
        odd_valid = self._odd_cells[shots_1d[self._odd_cells] == UNSHOT]
        if odd_valid.size > 0:
            if self.randomize_hunt:
                idx = int(self.rng.integers(0, odd_valid.size))
                return int(odd_valid[idx])
            return int(odd_valid[0])

        raise ValueError("No quedan casillas válidas por atacar.")


def evaluate_baseline_bot(
    bot: BaselineBot,
    fixed_ship_ids: np.ndarray,
) -> np.ndarray:
    """Ejecuta un bot de referencia sobre un lote fijo de flotas y devuelve n_shots (N,)."""
    n_games = fixed_ship_ids.shape[0]
    env = BattleshipVecEnv(batch_size=n_games)
    env.reset(fixed_ship_ids=fixed_ship_ids)

    for _ in range(NUM_CELLS):
        if np.all(env.done):
            break
        actions = np.zeros((n_games,), dtype=np.int64)
        for b in np.flatnonzero(~env.done):
            actions[b] = bot.select_action(env.shots[b])
        env.step(actions)

    return env.n_shots.copy()

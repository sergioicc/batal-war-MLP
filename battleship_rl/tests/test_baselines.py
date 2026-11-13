"""Prueba 5: Funcionamiento del Bot Aleatorio y Bot Heurístico Sencillo (Hunt & Target)."""

import numpy as np

from battleship_rl.agents.baselines import RandomBot, SimpleHeuristicBot, evaluate_baseline_bot
from battleship_rl.game.constants import BOARD_SIZE, HIT, UNSHOT
from battleship_rl.game.engine_ref import BattleshipRefEnv
from battleship_rl.utils.metrics import generate_eval_fleets


def test_05_random_and_heuristic_hunt_target_bots() -> None:
    """Prueba 5: RandomBot no repite; SimpleHeuristicBot sigue paridad en Hunt y caza en Target."""
    # 1. Verificar modo Hunt (paridad 2) y Target (vecinos ortogonales) paso a paso
    bot = SimpleHeuristicBot(seed=10, randomize_hunt=False)
    env = BattleshipRefEnv(seed=10)
    seen_actions = set()

    while not env.done:
        has_active_hit = bool(np.any(env.shots == HIT))
        even_remaining = any(
            env.shots[r * BOARD_SIZE + c] == UNSHOT
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if (r + c) % 2 == 0
        )
        action = bot.select_action(env.shots)
        assert action not in seen_actions
        seen_actions.add(action)

        r, c = divmod(action, BOARD_SIZE)
        if not has_active_hit and even_remaining:
            assert (r + c) % 2 == 0, "En modo Hunt debe respetar patrón cada dos espacios."

        if has_active_hit:
            # Debe ser vecino ortogonal de al menos un HIT activo
            active_hits = np.flatnonzero(env.shots == HIT)
            is_adjacent = any(
                abs(r - (h // BOARD_SIZE)) + abs(c - (h % BOARD_SIZE)) == 1
                for h in active_hits
            )
            assert is_adjacent, "En modo Target debe atacar un vecino ortogonal de un HIT activo."

        env.step(action)

    assert 17 <= env.n_shots <= 100

    # 2. Comparar ambos bots sobre 40 mapas idénticos: Heurístico Hunt&Target << Aleatorio
    fleets = generate_eval_fleets(num_games=40, seed=555)
    random_shots = evaluate_baseline_bot(RandomBot(seed=1), fleets)
    heur_shots = evaluate_baseline_bot(SimpleHeuristicBot(seed=2, randomize_hunt=True), fleets)

    assert np.all((random_shots >= 17) & (random_shots <= 100))
    assert np.all((heur_shots >= 17) & (heur_shots <= 100))
    assert float(np.mean(heur_shots)) < float(np.mean(random_shots)) - 15.0

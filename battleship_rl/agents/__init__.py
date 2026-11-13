"""Agentes: red neuronal MLP y bots de referencia (baselines)."""

from battleship_rl.agents.baselines import RandomBot, SimpleHeuristicBot, evaluate_baseline_bot
from battleship_rl.agents.mlp_agent import MLPPolicyNet, MLPAgent

__all__ = [
    "MLPPolicyNet",
    "MLPAgent",
    "RandomBot",
    "SimpleHeuristicBot",
    "evaluate_baseline_bot",
]

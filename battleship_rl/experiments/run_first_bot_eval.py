"""Experimento listo para ejecutar: entrenamiento de un primer bot MLP y comparativa de resultados.

Uso recomendado:
    python -m battleship_rl.experiments.run_first_bot_eval --blocks 100 --eval-games 100
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from battleship_rl.config import Config
from battleship_rl.learning.trainer import DualScaleReinforceTrainer
from battleship_rl.utils.metrics import compare_agents


def _format_comparison_table(
    before: Dict[str, Dict[str, float]],
    after: Dict[str, Dict[str, float]],
) -> str:
    rows = [
        ("MLP (sin entrenar)", before["mlp"]),
        ("MLP (entrenado)", after["mlp"]),
        ("Bot Aleatorio", after["random"]),
        ("Bot Heurístico (Hunt & Target)", after["heuristic"]),
    ]
    header = f"{'Agente':<34} | {'Media':>7} | {'Std':>7} | {'Min':>5} | {'Max':>5}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for name, st in rows:
        lines.append(
            f"{name:<34} | {st['mean']:7.2f} | {st['std']:7.2f} | {st['min']:5.0f} | {st['max']:5.0f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _plot_results(
    train_df: pd.DataFrame,
    before: Dict[str, Dict[str, float]],
    after: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Curva de aprendizaje por bloque de 20 rondas
    ax1 = axes[0]
    blocks = train_df["block"]
    means = train_df["mean_shots"]
    stds = train_df["std_shots"]
    ax1.plot(blocks, means, label="MLP media por bloque (20 rondas)", color="#1f77b4", linewidth=2)
    ax1.fill_between(
        blocks,
        means - stds,
        means + stds,
        alpha=0.2,
        color="#1f77b4",
        label="± 1 desviación estándar",
    )
    ax1.axhline(
        after["random"]["mean"],
        color="#d62728",
        linestyle="--",
        label=f"Bot Aleatorio ({after['random']['mean']:.1f})",
    )
    ax1.axhline(
        after["heuristic"]["mean"],
        color="#2ca02c",
        linestyle="-.",
        label=f"Heurístico Hunt & Target ({after['heuristic']['mean']:.1f})",
    )
    ax1.set_title("Curva de aprendizaje del primer bot MLP")
    ax1.set_xlabel("Bloque (20 rondas cada uno)")
    ax1.set_ylabel("Disparos promedio para ganar")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Comparativa final sobre los mismos mapas de evaluación
    ax2 = axes[1]
    labels = ["MLP\n(inicial)", "MLP\n(entrenado)", "Bot\nAleatorio", "Heurístico\n(Hunt & Target)"]
    vals = [
        before["mlp"]["mean"],
        after["mlp"]["mean"],
        after["random"]["mean"],
        after["heuristic"]["mean"],
    ]
    errs = [
        before["mlp"]["std"],
        after["mlp"]["std"],
        after["random"]["std"],
        after["heuristic"]["std"],
    ]
    colors = ["#9ecae1", "#1f77b4", "#d62728", "#2ca02c"]
    ax2.bar(labels, vals, yerr=errs, capsize=5, color=colors, alpha=0.9)
    ax2.set_title("Resultados finales en mapas fijos de evaluación")
    ax2.set_ylabel("Disparos totales (menor es mejor)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un primer bot MLP y compara sus resultados contra los bots de referencia."
    )
    parser.add_argument("--blocks", type=int, default=100, help="Cantidad de bloques de 20 rondas.")
    parser.add_argument("--eval-games", type=int, default=100, help="Partidas fijas de evaluación.")
    parser.add_argument("--eval-every", type=int, default=10, help="Frecuencia de evaluación en bloques.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible.")
    parser.add_argument("--lr", type=float, default=0.1, help="Tasa de aprendizaje única.")
    parser.add_argument("--greedy", action="store_true", help="Usar evaluación codiciosa (argmax).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/first_bot",
        help="Directorio donde guardar CSVs, checkpoints y gráficas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        seed=args.seed,
        learning_rate=args.lr,
        num_blocks=args.blocks,
        eval_every_blocks=args.eval_every,
        eval_num_games=args.eval_games,
        eval_greedy=args.greedy,
        save_every_blocks=max(1, min(25, args.blocks)),
        checkpoint_dir=str(out_dir / "checkpoints"),
    )

    trainer = DualScaleReinforceTrainer(cfg)

    print(f"Dispositivo activo: {trainer.agent.device}")
    print(f"Evaluación inicial antes de entrenar ({cfg.eval_num_games} mapas fijos)...")
    before = compare_agents(
        trainer.agent,
        num_games=cfg.eval_num_games,
        eval_seed=cfg.eval_seed_base,
        greedy=cfg.eval_greedy,
    )

    print(f"\nIniciando entrenamiento: {cfg.num_blocks} bloques x {cfg.block_size} rondas...")
    history = trainer.train(
        num_blocks=cfg.num_blocks,
        checkpoint_dir=Path(cfg.checkpoint_dir),
        verbose=True,
    )

    print(f"\nEvaluación final después de entrenar ({cfg.eval_num_games} mapas fijos)...")
    after = compare_agents(
        trainer.agent,
        num_games=cfg.eval_num_games,
        eval_seed=cfg.eval_seed_base,
        greedy=cfg.eval_greedy,
    )

    print("\n=== RESULTADOS COMPARATIVOS DEL PRIMER BOT ===")
    print(_format_comparison_table(before, after))

    delta_vs_init = before["mlp"]["mean"] - after["mlp"]["mean"]
    delta_vs_rand = after["random"]["mean"] - after["mlp"]["mean"]
    delta_vs_heur = after["mlp"]["mean"] - after["heuristic"]["mean"]
    print(f"\nMejora vs MLP sin entrenar: {delta_vs_init:+.2f} disparos")
    print(f"Ventaja vs Bot Aleatorio:   {delta_vs_rand:+.2f} disparos")
    print(f"Brecha vs Heurístico H&T:   {delta_vs_heur:+.2f} disparos (<= 0 indica superarlo)")

    summary = {
        "config": {
            "blocks": cfg.num_blocks,
            "block_size": cfg.block_size,
            "eval_games": cfg.eval_num_games,
            "seed": cfg.seed,
            "greedy": cfg.eval_greedy,
        },
        "before_training": before,
        "after_training": after,
    }
    summary_path = out_dir / "first_bot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    train_df = pd.DataFrame(history["training_history"])
    plot_path = out_dir / "first_bot_results.png"
    _plot_results(train_df, before, after, plot_path)

    print(f"\nArchivos generados en: {out_dir.resolve()}")
    print(f"- Resumen JSON: {summary_path.name}")
    print(f"- Gráfica:      {plot_path.name}")
    print(f"- Checkpoints:  {(Path(cfg.checkpoint_dir)).resolve()}")


if __name__ == "__main__":
    main()

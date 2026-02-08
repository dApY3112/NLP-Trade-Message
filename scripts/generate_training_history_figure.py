import argparse
import json
from pathlib import Path


def _find_trainer_state_json(models_dir: Path, run_tag: str) -> Path | None:
    # Common layout: models/bert_models/<run_tag>_finetuned/checkpoint-*/trainer_state.json
    candidates = sorted(
        models_dir.glob(f"{run_tag}_finetuned/**/trainer_state.json"),
        key=lambda p: str(p),
    )
    if candidates:
        # Prefer the last checkpoint (highest checkpoint number if present).
        def ckpt_key(path: Path) -> int:
            for part in reversed(path.parts):
                if part.startswith("checkpoint-"):
                    try:
                        return int(part.split("checkpoint-")[-1])
                    except ValueError:
                        return -1
            return -1

        candidates.sort(key=ckpt_key)
        return candidates[-1]

    # Fallback: any trainer_state.json under models_dir that mentions the run_tag in its path
    any_states = list(models_dir.glob("**/trainer_state.json"))
    for p in any_states:
        if run_tag in str(p):
            return p

    return None


def _extract_series(log_history: list[dict]):
    train_points = []
    eval_points = []

    for row in log_history:
        epoch = row.get("epoch")
        step = row.get("step")

        if "loss" in row and "eval_loss" not in row:
            train_points.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "loss": row.get("loss"),
                    "learning_rate": row.get("learning_rate"),
                    "grad_norm": row.get("grad_norm"),
                }
            )

        if "eval_loss" in row or "eval_f1" in row:
            eval_points.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "eval_loss": row.get("eval_loss"),
                    "eval_f1": row.get("eval_f1"),
                    "eval_accuracy": row.get("eval_accuracy"),
                }
            )

    # Sort for plotting
    train_points.sort(key=lambda d: (d["step"] is None, d["step"] or 0))
    eval_points.sort(key=lambda d: (d["epoch"] is None, d["epoch"] or 0))
    return train_points, eval_points


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True, help="Run tag, e.g. distilroberta_seed42_... (without _final suffix)")
    parser.add_argument(
        "--models-dir",
        default=str(Path("models") / "bert_models"),
        help="Path to models/bert_models",
    )
    parser.add_argument(
        "--out",
        default=str(Path("paper_figures") / "fig8_training_history.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    trainer_state_path = _find_trainer_state_json(models_dir, args.run_tag)
    if trainer_state_path is None:
        raise FileNotFoundError(
            f"Could not find trainer_state.json for run_tag='{args.run_tag}'. "
            f"Expected under: {models_dir / (args.run_tag + '_finetuned')}"
        )

    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    log_history = state.get("log_history", [])
    train_points, eval_points = _extract_series(log_history)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200)
    fig.suptitle(f"Training history (run_tag: {args.run_tag})")

    # Left: training loss vs step
    ax = axes[0]
    if train_points:
        xs = [p["step"] for p in train_points]
        ys = [p["loss"] for p in train_points]
        ax.plot(xs, ys, marker="o", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Train loss")
        ax.set_title("Train loss")
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No train-loss logs found", ha="center", va="center")

    # Right: eval loss and eval f1 vs epoch
    ax = axes[1]
    if eval_points:
        epochs = [p["epoch"] for p in eval_points]
        eval_loss = [p["eval_loss"] for p in eval_points]
        eval_f1 = [p["eval_f1"] for p in eval_points]

        ax.plot(epochs, eval_loss, marker="o", label="Eval loss", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Eval loss")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(epochs, eval_f1, marker="s", color="#d62728", label="Eval F1", linewidth=1.5)
        ax2.set_ylabel("Eval F1")

        ax.set_title("Eval loss & F1")

        # Combined legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best", frameon=False)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No eval logs found", ha="center", va="center")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    print(f"Saved: {out_path}")
    print(f"Source: {trainer_state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

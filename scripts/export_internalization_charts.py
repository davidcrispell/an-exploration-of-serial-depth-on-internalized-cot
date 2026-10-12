#!/usr/bin/env python3
"""Export static Matplotlib charts from the archived multiplication dashboard."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = {
    "3x3": {"cot_tokens": 25, "deep_test": 99.2, "shallow_test": 98.8},
    "4x4": {"cot_tokens": 47, "deep_test": 91.9, "shallow_test": 92.3},
    "5x5": {"cot_tokens": 75, "deep_test": 95.8, "shallow_test": 0.3},
}

COLORS = {"deep": "#0072B2", "shallow": "#D55E00"}
MARKERS = {"deep": "o", "shallow": "s"}
LABELS = {"deep": "Deep 1 (12 layers)", "shallow": "Shallow 1 (6 layers)"}


def read_csv(path: Path) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | int | None] = {}
            for key, value in raw.items():
                if value in (None, ""):
                    row[key] = None
                elif key in {"epoch", "step", "removed", "correct", "total"}:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def moving_average(values: np.ndarray, window: int = 7) -> np.ndarray:
    if len(values) < window:
        return values
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def save_figure(
    fig: plt.Figure, output_dir: Path, stem: str, *, close: bool = True
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    if close:
        plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_loss_grid(
    source_dir: Path, output_dir: Path, kind: str, *, close: bool = True
) -> None:
    title = (
        "Explicit-CoT fine-tuning loss"
        if kind == "explicit"
        else "CoT internalization rollout loss"
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.25), sharey=False)
    for ax, (task, meta) in zip(axes, TASKS.items()):
        model_rows: dict[str, list[dict[str, float | int | None]]] = {}
        for model in ("deep", "shallow"):
            rows = read_csv(source_dir / f"{model}-{task}-{kind}-loss.csv")
            model_rows[model] = rows
            steps = np.array([row["step"] for row in rows], dtype=float)
            losses = np.array([row["loss"] for row in rows], dtype=float)
            ax.plot(steps, losses, color=COLORS[model], alpha=0.16, linewidth=0.7)
            ax.plot(
                steps,
                moving_average(losses),
                color=COLORS[model],
                linewidth=1.8,
                label=LABELS[model],
            )

        if kind == "internalization":
            boundaries: list[dict[str, float | int | None]] = []
            previous_epoch = None
            for row in model_rows["deep"]:
                if row["epoch"] != previous_epoch:
                    boundaries.append(row)
                    previous_epoch = row["epoch"]
            for row in boundaries[1:]:
                ax.axvline(float(row["step"]), color="#999999", linewidth=0.55, alpha=0.45)

        ax.set_yscale("log")
        ax.set_title(f"{task} multiplication", fontsize=12, fontweight="semibold")
        ax.set_xlabel("Optimizer step")
        style_axis(ax)

    axes[0].set_ylabel("Cross-entropy loss (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=15, fontweight="semibold", y=1.09)
    fig.text(
        0.5,
        1.01,
        "Thin lines: logged batches · Thick lines: 7-point moving average",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, output_dir, f"{kind}-loss-curves", close=close)


def checkpoint_labels(rows: list[dict[str, float | int | None]], total: int) -> list[str]:
    labels = ["Explicit"]
    full_seen = 0
    for row in rows:
        removed = int(row["removed"] or 0)
        if removed < total:
            labels.append(str(removed))
        elif full_seen == 0:
            labels.append(f"{total}/{total}")
            full_seen += 1
        else:
            labels.append(f"A{full_seen}")
            full_seen += 1
    return labels


def plot_validation_grid(
    source_dir: Path, output_dir: Path, *, close: bool = True
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.75), sharey=True)
    for ax, (task, meta) in zip(axes, TASKS.items()):
        rows_by_model: dict[str, list[dict[str, float | int | None]]] = {}
        for model in ("deep", "shallow"):
            rows = read_csv(source_dir / f"{model}-{task}-internalization-validation.csv")
            rows_by_model[model] = rows
            values = [100.0] + [float(row["accuracy"] or 0.0) for row in rows]
            ax.plot(
                np.arange(len(values)),
                values,
                color=COLORS[model],
                linewidth=2.1,
                marker=MARKERS[model],
                markersize=5.0,
                label=LABELS[model],
            )

        labels = checkpoint_labels(rows_by_model["deep"], meta["cot_tokens"])
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        ax.axhline(99, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_ylim(-3, 104)
        ax.set_title(f"{task} multiplication", fontsize=12, fontweight="semibold")
        ax.set_xlabel("Curriculum checkpoint")
        ax.text(
            0.02,
            0.035,
            f"Final test: Deep {meta['deep_test']:.1f}% · Shallow {meta['shallow_test']:.1f}%",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#444444",
        )
        style_axis(ax)

    axes[0].set_ylabel("Validation exact-match accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Accuracy as visible CoT tokens are removed", fontsize=15, fontweight="semibold", y=1.09)
    fig.text(
        0.5,
        1.01,
        "The first full/full checkpoint is answer-only; A1+ are additional full-removal adaptation epochs.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "validation-accuracy-curves", close=close)


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "artifacts/dashboards/deep1-5x5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results/multiplication/figures",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Keep the figures open in native Matplotlib windows.",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    plot_loss_grid(args.source_dir, args.output_dir, "explicit", close=not args.show)
    plot_loss_grid(args.source_dir, args.output_dir, "internalization", close=not args.show)
    plot_validation_grid(args.source_dir, args.output_dir, close=not args.show)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

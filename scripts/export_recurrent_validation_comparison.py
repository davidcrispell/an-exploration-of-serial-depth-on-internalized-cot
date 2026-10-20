#!/usr/bin/env python3
"""Plot the live paper-recurrent validation curve against completed 5x5 runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "multiplication"
OUTPUT = RESULTS / "figures"


def load(name: str):
    with (RESULTS / name).open() as handle:
        return json.load(handle)


def through_first_full(rows: list[dict], removed_key: str) -> list[dict]:
    kept = []
    for row in rows:
        kept.append(row)
        if int(row[removed_key]) == 75:
            break
    return kept


def historical_xy(
    rows: list[dict], removed_key: str, accuracy_key: str, explicit_accuracy: float
) -> tuple[list[int], list[float]]:
    rows = through_first_full(rows, removed_key)
    return (
        [0] + [int(row[removed_key]) for row in rows],
        [100 * explicit_accuracy] + [100 * float(row[accuracy_key]) for row in rows],
    )


def current_xy(current: dict, recurrence: int) -> tuple[list[int], list[float]]:
    warmup = next(
        row for row in current["warmup"]["validation"] if row["recurrence"] == recurrence
    )
    points = []
    for epoch in current["internalization"]:
        validation = next(
            row for row in epoch["validation"] if row["recurrence"] == recurrence
        )
        points.append((int(epoch["removed_tokens"]), 100 * float(validation["accuracy"])))
    return [0] + [x for x, _ in points], [100 * warmup["accuracy"]] + [y for _, y in points]


def style_axis(axis: plt.Axes) -> None:
    axis.set_xlim(-2, 78)
    axis.set_ylim(-3, 103)
    axis.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 75])
    axis.set_yticks([0, 20, 40, 60, 80, 100])
    axis.grid(True, alpha=0.24, linewidth=0.8)
    axis.axhline(99, color="#999999", linestyle=":", linewidth=1.2, label="99% gate")
    axis.set_xlabel("Removed CoT tokens (of 75)")


def main() -> None:
    shallow = load("shallow-5x5-internalized-cot-progress.json")["completed_checkpoints"]
    deep = load("deep1-5x5-internalized-cot-validation.json")["validation_curve"]
    old_recurrent = [
        row
        for row in load("shallow1-1-recurrent-5x5-internalized-cot-progress.json")
        if row.get("stage") == "internalization"
    ]
    current = load("shallow1-1-paper-recurrent-5x5-progress.json")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharey=True)

    historical = [
        ("Shallow 1 (6 layers)", historical_xy(shallow, "removed_tokens", "accuracy", 1.0), "#D55E00", "s", "-"),
        ("Deep 1 (12 layers)", historical_xy(deep, "scheduled_removal", "accuracy", 1.0), "#0072B2", "o", "-"),
        ("Old recurrent pilot (frozen stack ×2)", historical_xy(old_recurrent, "scheduled_removal", "accuracy", 0.994), "#666666", "^", "--"),
        ("Paper-recurrent Shallow 1.1 (r=4)", current_xy(current, 4), "#7E57C2", "D", "-"),
    ]
    for label, (x_values, y_values), color, marker, linestyle in historical:
        axes[0].plot(
            x_values,
            y_values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.5 if "Paper-recurrent" in label else 1.8,
            markersize=5.5,
        )
    style_axis(axes[0])
    axes[0].set_ylabel("Validation exact-match accuracy (%)")
    axes[0].set_title("Primary recurrence vs past 5×5 runs")
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.92)

    recurrence_styles = [
        (1, "#D55E00", "s"),
        (2, "#009E73", "^"),
        (4, "#7E57C2", "D"),
        (8, "#0072B2", "o"),
    ]
    for recurrence, color, marker in recurrence_styles:
        x_values, y_values = current_xy(current, recurrence)
        effective_layers = next(
            row["effective_layers"]
            for row in current["warmup"]["validation"]
            if row["recurrence"] == recurrence
        )
        axes[1].plot(
            x_values,
            y_values,
            label=f"r={recurrence} ({effective_layers} effective layers)",
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=5.5,
        )
    style_axis(axes[1])
    axes[1].set_title("Paper-recurrent validation by recurrence")
    axes[1].legend(loc="lower left", frameon=True, framealpha=0.92)

    figure.suptitle("5×5 CoT internalization validation", fontsize=15, y=1.01)
    figure.text(
        0.5,
        0.005,
        "Current run through 48/75 removed tokens; historical curves stop at their first full-removal checkpoint.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "paper-recurrent-validation-comparison"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()

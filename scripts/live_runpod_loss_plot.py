#!/usr/bin/env python3
"""Show the active Runpod training loss in a live Matplotlib window."""

from __future__ import annotations

import argparse
import threading

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from live_runpod_loss_dashboard import State


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--refresh", type=int, default=10)
    parser.add_argument("--lines", type=int, default=10000)
    args = parser.parse_args()

    state = State(args)
    threading.Thread(target=state.refresh_forever, daemon=True).start()

    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(11, 6.5))
    figure.canvas.manager.set_window_title("Deep 1 · live training loss")

    def update(_: int) -> None:
        with state.lock:
            payload = dict(state.payload)
        stages = payload.get("stages", [])
        if not stages:
            axis.clear()
            axis.set_title("Waiting for the first log refresh…")
            return

        stage = stages[-1]
        points = stage.get("points", [])
        if not points:
            return

        x_values = [index * 100 for index in range(1, len(points) + 1)]
        losses = [point["loss"] for point in points]
        smoothing = max(1, min(15, len(losses) // 20))
        ema: list[float] = []
        alpha = 2 / (smoothing + 1)
        for loss in losses:
            ema.append(loss if not ema else alpha * loss + (1 - alpha) * ema[-1])

        axis.clear()
        axis.set_yscale("log")
        axis.plot(x_values, losses, color="#54d6ff", alpha=0.32, linewidth=1, label="reported loss")
        axis.plot(x_values, ema, color="#54d6ff", linewidth=2.2, label=f"EMA ({smoothing} samples)")

        previous = points[0].get("removed")
        for index, point in enumerate(points[1:], start=1):
            removed = point.get("removed")
            if removed is not None and removed != previous:
                axis.axvline(x_values[index], color="#ffb454", alpha=0.30, linewidth=0.8)
                previous = removed

        validation_text = ""
        validations = stage.get("validations", [])
        if validations:
            last_validation = validations[-1]
            validation_text = f" · validation {last_validation['accuracy']:.1f}%"

        latest = points[-1]
        removal_text = "" if latest.get("removed") is None else f" · removed {latest['removed']}"
        axis.set_title(
            stage["name"].replace("_", " ")
            + f"\nloss {latest['loss']:.5g}{removal_text}{validation_text}"
        )
        axis.set_xlabel("Approximate optimizer steps (logged every 100)")
        axis.set_ylabel("Training loss (log scale)")
        axis.grid(alpha=0.16)
        axis.legend(loc="upper right")
        figure.tight_layout()

    animation = FuncAnimation(
        figure,
        update,
        interval=args.refresh * 1000,
        cache_frame_data=False,
    )
    figure._runpod_animation = animation
    plt.show()


if __name__ == "__main__":
    main()

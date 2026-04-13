"""Live training monitor — reads logs/training_log.csv and plots in real time."""

import matplotlib
matplotlib.use("MacOSX")

import time
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LOG_PATH = "logs/training_log.csv"
REFRESH_MS = 2000  # refresh every 2 seconds


def read_log():
    episodes, scores, eval_scores = [], [], []
    if not os.path.exists(LOG_PATH):
        return episodes, scores, eval_scores
    with open(LOG_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                episodes.append(int(row["episode"]))
                scores.append(int(row["score"]))
                eval_scores.append(float(row["eval_mean_score"]) if row["eval_mean_score"] else None)
            except (ValueError, KeyError):
                continue
    return episodes, scores, eval_scores


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
fig.suptitle("Snake RL — Live Training", fontsize=14, fontweight="bold")
plt.tight_layout(pad=3)


def update(_frame):
    episodes, scores, eval_scores = read_log()
    if not episodes:
        return

    # Rolling mean (window=50)
    window = 50
    rolling = []
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        rolling.append(sum(scores[start:i+1]) / (i - start + 1))

    # Eval points
    eval_eps   = [e for e, v in zip(episodes, eval_scores) if v is not None]
    eval_vals  = [v for v in eval_scores if v is not None]

    ax1.clear()
    ax1.plot(episodes, scores, color="#aaaaaa", alpha=0.4, linewidth=0.8, label="Episode score")
    ax1.plot(episodes, rolling, color="#3399ff", linewidth=2, label=f"Rolling mean ({window})")
    if eval_eps:
        ax1.plot(eval_eps, eval_vals, "o-", color="#ff6633", linewidth=2, markersize=5, label="Eval mean score")
        ax1.set_title(f"Scores  |  Latest eval: {eval_vals[-1]:.1f}  |  Best eval: {max(eval_vals):.1f}", fontsize=11)
    else:
        ax1.set_title("Scores", fontsize=11)
    ax1.set_ylabel("Score (food eaten)")
    ax1.set_xlabel("Episode")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Score distribution (last 200 episodes)
    recent = scores[-200:]
    ax2.clear()
    ax2.hist(recent, bins=30, color="#3399ff", edgecolor="white", alpha=0.8)
    ax2.set_title(f"Score distribution (last {len(recent)} episodes)  |  Max ever: {max(scores)}", fontsize=11)
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    fig.canvas.draw()


ani = animation.FuncAnimation(fig, update, interval=REFRESH_MS, cache_frame_data=False)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from env.gridWorldEnv import GridWorld
from algos.valueIteration import extract_policy, value_iteration


def plot_values(value_fn, env, title="Value function"):
    # For imshow: rows = y, cols = x, so we need (height, width) with [y,x] = value_fn[x,y]
    grid = value_fn.T

    fig, ax = plt.subplots()
    im = ax.imshow(
        grid,
        origin="lower",
        interpolation="none",
        cmap="viridis",
    )

    ax.set_xticks(np.arange(env.width))
    ax.set_yticks(np.arange(env.height))
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for x in range(env.width):
        for y in range(env.height):
            val = value_fn[x, y]
            ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="V(s)")
    plt.tight_layout()
    return fig, ax


def plot_policy(policy, env, title="Policy"):
    """Plot policy as a grid of action labels (e.g. ↑, →)."""
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(env.width))
    ax.set_yticks(np.arange(env.height))
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for x in range(env.width):
        for y in range(env.height):
            ax.text(x, y, str(policy[x, y]), ha="center", va="center", fontsize=12)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    env = GridWorld(8)
    V = value_iteration(env)

    plot_values(V, env, title="Value iteration – V(s)")

    policy = extract_policy(env, V)
    env.policy = policy

    print("Policy (action per state):")
    print(policy.T) 
    plot_policy(env.policy, env, title="Policy π(s)")

    plt.show()

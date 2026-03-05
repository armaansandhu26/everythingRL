from env.gridWorldEnv import GridWorld
from env.gridWorldVisualizer import GridWorldVisualizer
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":
    env = GridWorld(8)
    viz = GridWorldVisualizer(env)

    env.reset()
    viz.render("reset")

    dummy_actions = ["RIGHT"] * 4 + ["UP"] * 2 + ["END"]
    for action in dummy_actions:
        state, reward, done = env.step(action)
        viz.render(f"action={action}")
        time.sleep(0.5)
        if done:
            break

    viz.render("episode finished")

    plt.ioff()
    plt.show()
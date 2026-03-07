from env.gridWorldEnv import GridWorld
from env.gridWorldVisualizer import GridWorldVisualizer
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":
    env = GridWorld(8)
    viz = GridWorldVisualizer(env)

    # Use env.policy if set; otherwise compute it with value iteration
    if env.policy is None:
        from algos.valueIteration import value_iteration, extract_policy
        V = value_iteration(env)
        env.policy = extract_policy(env, V)

    env.reset()
    viz.render("reset")

    while not env.done:
        state = env.current_state
        action_label = env.get_action(state)
        if action_label == "T" or state in getattr(env, "terminal_states", set()):
            break
        action = env.policy_label_to_action(action_label)
        if action is None:
            break
        env.step(action)
        viz.render(f"action={action}")
        time.sleep(0.5)

    viz.render("episode finished")

    plt.ioff()
    plt.show()
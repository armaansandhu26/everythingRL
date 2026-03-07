import numpy as np

def value_iteration(env, max_iters=10000, threshold=1e-6):
    V = np.zeros([env.width, env.height])

    for iter in range(max_iters):
        v_new = V.copy()
        delta = 0
        for i in range(env.width):
            for j in range(env.height):
                best_val = -float("inf")
                for action in env.actions:
                    next_state, reward, done = env.transition((i, j), action)
                    if done:
                        candidate = reward
                    else:
                        candidate = reward + env.gamma * V[next_state[0]][next_state[1]]

                    best_val = max(best_val, candidate)
                v_new[i][j] = best_val
                delta = max(delta, abs(v_new[i][j] - V[i][j]))

        V = v_new
        if iter % 10 == 0:
            print("value iteration - ", iter, ": delta - ", delta, "\n")

        if delta < threshold:
            print("converged at iteration", iter)
            break

    return V

def extract_policy(env, V):
    policy = np.empty((env.width, env.height), dtype=object)
    for i in range(env.width):
        for j in range(env.height):
            state = (i, j)
            if state in getattr(env, "terminal_states", set()):
                policy[i][j] = env.get_action_icon("T") if hasattr(env, "get_action_icon") else "T"
                continue

            best_action = None
            best_value = -float("inf")

            for action in env.actions:
                next_state, reward, done = env.transition(state, action)
                if done:
                    value = reward
                else:
                    value = reward + env.gamma * V[next_state[0]][next_state[1]]

                if value > best_value:
                    best_value = value
                    best_action = action

            policy[i][j] = env.get_action_icon(best_action)

    return policy

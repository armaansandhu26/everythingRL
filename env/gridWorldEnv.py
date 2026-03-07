class GridWorld:

    def __init__(self, size=8):
        self.width = size
        self.height = size

        self.start = (0, 0)
        self.gamma = 0.95

        self.actions = ["UP", "RIGHT"]

        self.rewards = {
            (0, 2): 1,
            (2, 0): 1,
            (0, 4): 2,
            (4, 0): 2,
            (0, 6): 3,
            (6, 0): 3,

            (2, 2): 2,
            (3, 3): 4,
            (4, 4): -2,

            (7, 7): 10,

            (7, 0): -3,
            (0, 7): -3,
        }

        self.terminal_states = {
            (7, 7),
        }
        self.current_state = self.start
        self.done = False
        self.total_reward = 0

        self.policy = None

        self._policy_label_to_action = {
            "↑": "UP", 
            "→": "RIGHT", 
            "↓": "DOWN", 
            "←": "LEFT", 
            "UP": "UP", 
            "RIGHT": "RIGHT", 
            "DOWN": "DOWN", 
            "LEFT": "LEFT"}


    def step(self, action):
        if self.done:
            return self.current_state, 0, True

        x, y = self.current_state
        if action == "UP":
            next_state = (x, min(y + 1, self.height - 1))
        elif action == "RIGHT":
            next_state = (min(x + 1, self.width - 1), y)
        else:
            raise ValueError(f"Invalid action: {action}")

        reward = self.rewards.get(next_state, 0)
        self.total_reward += reward
        if next_state in self.terminal_states:
            self.done = True

        self.current_state = next_state

        return next_state, reward, self.done


    def transition(self, state, action):

        x, y = state
        if action == "UP":
            next_state = (x, min(y + 1, self.height - 1))
        elif action == "RIGHT":
            next_state = (min(x + 1, self.width - 1), y)
        else:
            raise ValueError(f"Invalid action: {action}")

        reward = self.rewards.get(next_state, 0)
        done = next_state in self.terminal_states

        return next_state, reward, done


    def reset(self):
        self.current_state = self.start
        self.done = False
        self.total_reward = 0
        return self.current_state


    def get_states(self):
        return [(x, y) for x in range(self.width) for y in range(self.height)]


    def get_actions(self):
        return self.actions

    def get_action_icon(self, action):
        actions_to_icons = {
            "UP": "↑",
            "RIGHT": "→",
            "T": "T", 
            None: " "
        }

        return actions_to_icons.get(action, "?")

    def get_action(self, state):
        if self.policy is None:
            return None
        x, y = state
        return self.policy[x, y]

    def policy_label_to_action(self, label):
        if label is None:
            return None
        key = str(label).strip()
        if key == "T":
            return None
        return self._policy_label_to_action.get(key)
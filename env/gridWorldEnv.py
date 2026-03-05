
class GridWorld:
    def __init__(self, size = 8):
        self.width = size
        self.height = size
        self.start = (0, 0)
        self.actions = ['UP', 'RIGHT', 'END']
        self.rewards = {
            (0, 4): 1,
            (4, 0): 1,
            (0, 7): -1,
            (7, 0): -1,
        }
        self.total_reward = 0
        self.current_state = self.start
        self.done = False

    def step(self, action):
        if self.done:
            return self.current_state, 0, self.done

        if action == 'END':
            self.done = True
            return self.current_state, self.total_reward, self.done
        elif action == 'UP':
            self.current_state = (self.current_state[0], min(self.current_state[1]+1, self.height-1))
        elif action == 'RIGHT':
            self.current_state = (min(self.current_state[0]+1, self.width-1), self.current_state[1])
        self.total_reward += self.rewards.get(self.current_state, 0)
  
        return self.current_state, 0, self.done

    def reset(self):
        self.current_state = self.start
        self.total_reward = 0
        self.done = False
        return self.current_state
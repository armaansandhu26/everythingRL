import numpy as np
import matplotlib.pyplot as plt

class GridWorldVisualizer:
    def __init__(self, env, cell_size=1.0):
        """
        env must have: width, height, start, rewards, current_state
        """
        self.env = env
        self.cell_size = cell_size

        # interactive mode for real-time updates
        plt.ion()

        self.fig, self.ax = plt.subplots()
        self._initialized = False

        self.im = None
        self.agent_scatter = None
        self.text_handles = []
        self.status_text = None

    def _build_background(self):
        """
        Create a background matrix:
          0 for empty
          +1 for positive reward cells
          -1 for negative reward cells
        """
        bg = np.zeros((self.env.height, self.env.width), dtype=np.float32)
        for (x, y), r in self.env.rewards.items():
            if 0 <= x < self.env.width and 0 <= y < self.env.height:
                bg[y, x] = 1.0 if r > 0 else -1.0
        return bg

    def _init_plot(self):
        self.ax.clear()

        bg = self._build_background()

        # show background
        self.im = self.ax.imshow(
            bg,
            origin="lower",  # (0,0) at bottom-left, y increases upward
            interpolation="none",
            vmin=-1,
            vmax=1,
        )

        # grid lines
        self.ax.set_xticks(np.arange(-0.5, self.env.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.env.height, 1), minor=True)
        self.ax.grid(which="minor")
        self.ax.tick_params(which="minor", bottom=False, left=False)

        self.ax.set_xticks(np.arange(self.env.width))
        self.ax.set_yticks(np.arange(self.env.height))
        self.ax.set_xlim(-0.5, self.env.width - 0.5)
        self.ax.set_ylim(-0.5, self.env.height - 0.5)

        # reward text labels
        self.text_handles = []
        for (x, y), r in self.env.rewards.items():
            if 0 <= x < self.env.width and 0 <= y < self.env.height:
                t = self.ax.text(x, y, str(r), ha="center", va="center")
                self.text_handles.append(t)

        # start marker
        sx, sy = self.env.start
        self.ax.text(sx, sy, "S", ha="center", va="center")

        # agent marker
        ax_, ay_ = self.env.current_state
        self.agent_scatter = self.ax.scatter([ax_], [ay_], s=150, marker="o")

        # status text (state, total reward, done) at top-left of the figure
        self.status_text = self.fig.text(
            0.01,
            0.95,
            "",
            transform=self.fig.transFigure,
            fontsize=10,
            verticalalignment="top",
        )

        self._initialized = True
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self, title=None):
        if not self._initialized:
            self._init_plot()

        # update background in case rewards change
        bg = self._build_background()
        self.im.set_data(bg)

        # move agent
        ax_, ay_ = self.env.current_state
        self.agent_scatter.set_offsets(np.array([[ax_, ay_]], dtype=np.float32))

        # title/status
        if title is None:
            title = f"state={self.env.current_state}  total_reward={getattr(self.env, 'total_reward', None)}  done={getattr(self.env, 'done', None)}"
        self.ax.set_title(title)

        # overlay detailed status text on the figure
        if self.status_text is not None:
            status = f"state={self.env.current_state}  total_reward={getattr(self.env, 'total_reward', None)}  done={getattr(self.env, 'done', None)}"
            self.status_text.set_text(status)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # tiny pause to let UI update
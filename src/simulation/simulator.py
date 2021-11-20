import numpy as np
from collections import defaultdict
from typing import Type, List, Tuple, Any, Dict, Union, Set
import matplotlib.pyplot as plt

from .logger import Logger
from ..prism.strategy import Strategy, ActionSequenceStrategy, RandomStrategy
from ..graph.two_player_graph import TwoPlayerGraph
from ..graph.product import ProductAutomaton
from wombats.systems.minigrid import MultiAgentMiniGridEnv

Result = Any
Actions = None
TILE_SIZE = 128


# TODO: This class should be integrated in env or game
class Simulator:

    _results = []

    def __init__(self, env: MultiAgentMiniGridEnv, game: ProductAutomaton):
        self._env = env
        self._env_width = self._env.unwrapped.width
        self._env_height = self._env.unwrapped.height
        self._game = game
        self._logger = Logger()
        self.reset()

    def reset(self):
        self._results = []
        self._episode = 0
        self._logger.reset()

    def reset_episode(self):
        self._costs = []
        self._observations = []
        self._sys_actions = []
        self._env_actions = []
        self._sys_grid = np.zeros((self._env_width, self._env_height))
        self._env_grid = np.zeros((self._env_width, self._env_height))

    def run(self, iterations: int = 1, plot: bool = True, debug: bool = False,
        **kwargs) -> List[Result]:

        for iteration in range(iterations):

            if self._env.concurrent:
                self.run_concurrent_game(**kwargs)
            else:
                self.run_turn_based_game(**kwargs)

            if debug:
                self._logger.print_episode(self._episode)
            self._episode += 1

        # print(f'Average values for {iterations} runs')
        # stats = self._logger.get_stats()
        # print(stats)
        return self._results

    def get_stats(self):
        costs = np.max([r['Cost'] for r in self._results], axis=0)
        count_obs = defaultdict(lambda: 0)
        for r in self._results:
            count_obs[tuple(r['Observation'])] += 1

        print('Maximum Costs', costs)
        print('Observation', dict(count_obs))

    def plot_grid(self):

        self._grids = np.zeros((self._env_width, self._env_height))
        for r in self._results:
            self._grids += r['SysPlay']

        x = np.arange(self._env_width+1)
        y = np.arange(self._env_height+1)

        fig, ax = plt.subplots()

        ax.pcolormesh(y, x, np.transpose(self._grids))
        # ax.pcolormesh(y, x, self._grids.T)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid()
        plt.show()

    def run_concurrent_game(self) ->  Result:
        raise NotImplementedError('Not yet!')

    def run_turn_based_game(self, sys_strategy: Strategy = None, sys_actions: Actions = None,
                             env_strategy: Strategy = None, env_actions: Actions = None,
                             render: bool = True, record_video: bool = True, adversarial: bool = True):

        self.reset_episode()

        sys_strategy, env_strategy = self._check_strategies(
            sys_strategy, sys_actions, env_strategy, env_actions)

        sys_strategy.reset()
        self._env._toggle_video_recording(record_video=record_video)
        self._env.reset()
        self._simulate_on_env(render)
        env_state, env_action, sys_state, cost = self._game.reset() # None, None, Init

        self._costs.append(cost)
        pos = sys_state[0][1][0]
        self._sys_grid[pos[0], pos[1]] += 1

        done = False
        steps = 0

        while not done:

            # System's Action
            sys_action, env_state = sys_strategy.step(env_state, env_action, sys_state)
            # Use env_state that the system chose and not from the game.
            _, obs, cost, done = self._game.step(sys_state, sys_action, env_state)
            self.log(steps, cost, obs, sys_action, env_state, 'sys')
            self._simulate_on_env(render, sys_actions=sys_action)

            steps += 1

            if done:
                break

            # Environment's Action
            if adversarial:
                env_action, _ = env_strategy.step(sys_state, sys_action, env_state)
            else:
                env_action, _ = sys_strategy.step(sys_state, sys_action, env_state)

            sys_state, obs, cost, done = self._game.step(env_state, env_action)
            self.log(steps, cost, obs, env_action, sys_state, 'env')
            self._simulate_on_env(render, env_actions=env_action)

            steps += 1

        self._env._toggle_video_recording()
        self._env.close()

        self._results.append(self.get_episodic_data())

        if record_video:
            return self._env._get_video_path()
        return ''

    def _simulate_on_env(self, render, sys_actions=None, env_actions=None):
        if sys_actions is not None:
            for action in sys_actions:
                self._env.step([[action], [None]])

        if env_actions is not None:
            self._env.step([[None], *env_actions])

        if render:
            self._env.render()

    def _check_strategies(self, sys_strategy, sys_actions, env_strategy, env_actions):

        if sys_strategy is None and sys_actions is None:
            raise Exception('Either sys_strategy or sys_action must be specified')
        elif sys_actions is not None:
            sys_strategy = ActionSequenceStrategy(self._game, sys_actions)

        if env_strategy is None and env_actions is None:
            env_strategy = RandomStrategy(self._game)
        elif env_actions is not None:
            env_strategy = ActionSequenceStrategy(self._game, env_actions)

        return sys_strategy, env_strategy

    def log(self, steps, cost, obs, action, state, curr_player) -> None:
        """
        Log all metrics
        """
        self._logger.add_vector('Cost', cost, self._episode, steps)
        self._logger.add_strings('Observation', obs, self._episode, steps)
        self._logger.add_strings('SysPlay', state, self._episode, steps)

        if curr_player == 'sys':
            self._logger.add_strings('SysAction', action, self._episode, steps)
        else:
            self._logger.add_strings('EnvAction', action, self._episode, steps)

        obs = None if len(obs) == 0 else list(obs)[0]
        self._costs.append(cost)
        self._observations.append(obs)

        if curr_player == 'sys':
            self._sys_actions.append(action)
            pos = state[0][1][0]
            self._sys_grid[pos[0], pos[1]] += 1
        else:
            self._env_actions.append(action)
            # pos = state[0][1][0]
            # self._env_grid[pos[0], pos[1]] += 1

    def get_episodic_data(self):
        return {
            'Cost': np.sum(self._costs, axis=0),
            'Observation': [o for o in self._observations if o is not None],
            'SysActions': self._sys_actions,
            'EnvActions': self._env_actions,
            'SysPlay': self._sys_grid}


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Any, Union, List, Dict, Callable

Scalar = Union[int, float]
Vector = List[Scalar]
StringVector = List[str]
Result = Dict
Stats = Any


class Logger:

    _plot_funcs: defaultdict(lambda: Callable)

    def __init__(self):
        self.reset()

    def reset(self):
        self._scalars = None
        self._vectors = None
        self._strings = None
        self._string_vectors = None

    def add_scalar(self, name: str, data: Scalar, episode: int, steps: int) -> None:
        if data is None:
            return

        if not isinstance(data, Scalar):
            raise Exception(f'{data} is not a Scalar')

        if not isinstance(steps, int):
            raise Exception(f'{steps} is not a int')

        if self._scalars is None:
            self._scalars = defaultdict(lambda: defaultdict(lambda: []))

        self._scalars[name][episode].append(data)

    def add_vector(self, name: str, data: Vector, episode: int, steps: int) -> None:
        if data is None:
            return

        if not isinstance(data, List):
            raise Exception(f'{data} is not a Vector')

        if not isinstance(steps, int):
            raise Exception(f'{steps} is not a int')

        if self._vectors is None:
            self._vectors = defaultdict(lambda: defaultdict(lambda: []))

        self._vectors[name][episode].append(data)

    def add_string(self, name: str, data: str, episode: int, steps: int) -> None:
        if data is None:
            return

        if not isinstance(data, str):
            raise Exception(f'{data} is not a str')

        if not isinstance(steps, int):
            raise Exception(f'{steps} is not a int')

        if self._strings is None:
            self._strings = defaultdict(lambda: defaultdict(lambda: []))

        self._strings[name][episode].append(data)

    def add_strings(self, name: str, data: StringVector, episode: int, steps: int) -> None:
        if data is None:
            return

        # if not isinstance(data, List):
        #     raise Exception(f'{data} is not a StringVector')

        if not isinstance(steps, int):
            raise Exception(f'{steps} is not a int')

        if self._string_vectors is None:
            self._string_vectors = defaultdict(lambda: defaultdict(lambda: []))

        self._string_vectors[name][episode].append(data)

    def get_data(self):
        data = [self._scalars,
                self._vectors,
                self._strings,
                self._string_vectors,
                ]

        return [d for d in data if d is not None]

    def print_data(self):
        logs = self.get_data()
        for log in logs:
            for metric_name, episodic_data in log.items():
                print(metric_name)
                for episode, data in episodic_data.items():
                    print(episode, data)

    def print_episode(self, episode):
        logs = self.get_data()
        for log in logs:
            for metric_name, episodic_data in log.items():
                print(metric_name)
                print(episodic_data[episode])

    def get_stats(self) -> Stats:
        result = defaultdict(lambda: {})
        for name, episodic_data in self._vectors.items():

            data = list(episodic_data.values())
            take_sum = lambda x: np.sum(x, axis=0)
            data = list(map(take_sum, data))

            result[name]['mean'] = np.mean(data, axis=0)
            result[name]['std'] = np.std(data, axis=0)
            result[name]['max'] = np.max(data, axis=0)
            result[name]['min'] = np.min(data, axis=0)

        return dict(result)

    def plot(self,
             filedir: str = None,
             savefig: bool = True,
             extension: str = '.png',
             show: bool = True) -> None:

        if filedir is None:
            savefig = False

        for name, data in self._scalars.items():
            if name in self._plot_funcs:
                continue

            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(data)

            ax.set_xlabel('Steps')
            axs.set_ylabel(name)

            if savefig:
                filename = os.path.join(filedir, name + extension)
                plt.savefig(filename)

        for name, plot_func in self._plot_funcs:
            plot_func(name, self._scalars[name])

        if show:
            plt.show()

    def add_plot(self, name: str, plot_func: Callable) -> None:
        self._plot_funcs[name] = plot_func

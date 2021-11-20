import os
import re
import copy
import docker
import warnings
import paramiko
from base64 import decodebytes
import subprocess as sp
from pathlib import Path
from scipy.stats import rv_discrete
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional, Union, Hashable
import matplotlib.pyplot as plt

from ..graph import TwoPlayerGraph
# from ..strategy_synthesis import MultiObjectiveSolver
from .strategy import PrismMealyMachine, StochasticStrategy
from ..strategy_synthesis.adversarial_game import ReachabilityGame

from ..config import ROOT_PATH

Node = Hashable
Nodes = List[Node]
Action = str
Actions = List[Action]
ParetoPoint = Tuple[float, ...]
# ParetoPoint = str
Point = List[float]
Probability = float
DistOverPareto = Dict[ParetoPoint, Probability]


class PrismInterface():
    """
    Interface class to use PRISM either on local pc or on docker

    For installing prism locally, see https://www.prismmodelchecker.org/

    For Dockerfile, see https://github.com/aria-systems-group/prism-docker
    Build docker image with the command `docker build -t prism .`
    To run a docker container, first cd to the home directory of
    regret_synthesis_toolbox library and then run
    `run_docker.sh` located in the home directory.
    This will start a docker container and we can now send commands to this container
    using this PrismInterface class.

    If you change prism_binary to prismgames, you can also interact with prism-games
    """

    def __init__(self, prism_binary: str = 'prism',
                 use_docker: bool = False, image_name: str = 'prism', container_name: str = None,
                 hostname: str = None, port: int = 22, username: str = None, password: str = None,
                 local_dir: str=os.path.join(ROOT_PATH, 'prism'), remote_dir='/prism/configs'):
        """
        Tell PrismInterface where the prism binary is.
        Either
        1. If prism_binary is not specified => We will use local binary prism OR prismgames
        2. You can specify the binary location e.g.) `/SOMEWHERE/prism`
        3. You can choose to use prism-docker and access prism in a docker container
        4. Similarly you can use ssh to access a docker container
        You will need hostname (required), address, username, and password if you have any
        """
        self._prism_binary = prism_binary

        # Initially, we assume that we are running locally
        self._interface_method = 'local'
        self._local_dir = local_dir
        self._remote_dir = local_dir

        if use_docker:
            self._interface_method = 'docker'
            self._image_name = image_name
            # Check if such container exists
            if container_name:
                client = docker.from_env()
                cl = client.containers.list()
                container_names = [c.name for c in cl]
                if container_name not in container_names:
                    msg = f'There is no such container name as {container_name}'
                    raise ValueError(msg)
            self._container_name = container_name
            self._remote_dir = remote_dir
        if hostname:
            self._interface_method = 'ssh'
            self._client = paramiko.SSHClient()
            self._client.connect(hostname, port, username, password)
            self._remote_dir = remote_dir

    def run_prism(self, model_filename: str, prop_filename: str = None,
                  get_help: bool = False, **kwargs) -> str:
        """
        Call prism command
        - Model file is required
        - Property is required
            - as a file
            - as a command
        """
        cmd = self._get_command(**kwargs)

        if get_help:
            prism_call = self._prism_binary + ' ' + '-help'
        else:
            if prop_filename:
                prism_call = self._prism_binary + ' ' + model_filename + ' ' + prop_filename + ' ' + cmd
            else:
                if 'pctl' not in kwargs:
                    raise ValueError('Please provide either a property file or pctl formula')
                prism_call = self._prism_binary + ' ' + model_filename + ' ' + cmd

        completed_process = self._run_command(prism_call)

        return completed_process

    def _get_command(self, **kwargs) -> str:
        """
        gets a list of popt commands to send the binary

        :param      kwargs:  The flexfringe tool keyword arguments

        :returns:   The list of commands.
        """

        # default argument is to print the program's man page
        if(len(kwargs) > 1):
            cmds = []
            for key, value in kwargs.items():
                if isinstance(value, bool) and value:
                    c = '-' + key
                else:
                    c = '-' + key + ' ' + str(value)
                cmds.append(c)
            cmd = ' '.join(cmds)
        else:
            cmd = '-help'
            print('no options specified, printing tool help:')

        return cmd

    def _run_command(self, command_string: str):
        """
        Run provided command on either local, docker or ssh
        """
        if self._interface_method == 'local':
            completed_process = sp.run(command_string)
            return completed_process.stdout.decode()

        elif self._interface_method == 'docker':
            if self._container_name:
                container_name = self._container_name
            else:
                client = docker.from_env()
                cl = client.containers.list(filters={'ancestor': self._image_name})
                if len(cl) <= 0:
                    msg = f'There is no running container whose image name is {self._image_name}'
                    raise NameError(msg)
                elif len(cl) >= 2:
                    print(f'There are multiple containers with same image name. Please choose from the following option. Either specify the index or the container name\n')
                    choice = input('\n'.join([f'[{i}] {c.name}' for i, c in enumerate(cl)]))
                    include_number = lambda s: bool(re.search(r'\d', choice))
                    if include_number(choice):
                        index = int(choice)
                    else:
                        for i, c in enumerate(cl):
                            if c.name == choice:
                                index = i
                else:
                    index = 0

                container_name = cl[index].name

            command_string = f'docker exec -it {container_name} {command_string}'
            completed_process = sp.run(command_string, shell=True,
                                       stdout=sp.PIPE, stderr=sp.PIPE)
            return completed_process.stdout.decode()

        elif self._interface_method == 'ssh':
            stdin, stdout, stderr = self._client.exec_command(command_string)
            return stdout

        else:
            raise NotImplementedError('Choose either local or docker')


class PrismInterfaceForTwoPlayerGame(PrismInterface):

    def __init__(self, remote_dir='/prism-games/configs', **kwargs):
        super().__init__('prismgames', remote_dir=remote_dir, **kwargs)
        self.game: TwoPlayerGame = None
        self._game_to_prism_map: Dict = None
        self._prism_to_game_map: Dict = None
        self._sta_to_prism_state_map: Dict = None
        self._prism_action_idx_to_game_map: Dict = None
        self._game_str: Dict = None
        self._pareto_points = None
        self._initialized: bool = False
        self._max_weight_values: Dict = None

    def run_prism(self, game: TwoPlayerGraph, pareto_point = None, filename: str = None,
                  plot: bool = False, debug: bool = False, **kwargs) -> str:
        """
        Run prism games.
        Model and Property files are auto-generated from TwoPlayerGame
        It exports adversary and states files

        :args   filename:       Option to change default filename
        """
        # game = self.delete_loops(game)

        if pareto_point is not None:
            game = self._set_pareto_point(game, pareto_point)

        # Initialize mappings and export prism model & props files
        self._initialize_variables(game)

        # Prepare filenames for prism command
        if filename is None:
            filename = self.graph_name_in_prism

        extensions = ['.prism', '.props', '.adv', '.sta']

        fs = self._create_prism_filenames(extensions, self._remote_dir, filename)

        # Export adversary .adv file (strategy)
        if 'exportadv' not in kwargs:
            kwargs['exportstrat'] = fs['.adv']

        # Export .sta file to use for reading strategy
        if 'exportstates' not in kwargs:
            kwargs['exportstates'] = fs['.sta']

        # Run PRISM-Games
        if 'pctl' in kwargs:
            completed_process = super().run_prism(fs['.prism'], **kwargs)
        else:
            completed_process = super().run_prism(fs['.prism'], fs['.props'], **kwargs)

        if debug:
            for line in completed_process.split('\n'):
                print(line)

        # Postprocess (Analyze pareto points and Extract strategy)

        # If pareto was specified, find pareto_points from stdout
        if 'pareto' in kwargs:
            pareto_points = self._find_pareto_points_from_stdout(completed_process)
            self._pareto_points = pareto_points
            if plot:
                self._plot_pareto(pareto_points)

        # Read .sta & .adv files and convert to a strategy on TwoPlayerGame
        sta_local_path = os.path.join(self._local_dir, filename + '.sta')
        adv_local_path = os.path.join(self._local_dir, filename + '.adv')
        self._game_str = self._read_strategy(sta_local_path, adv_local_path)

        return completed_process

    def _initialize_variables(self, game):
        """
        Set a game (Somehow construction method gives me an error)
        """
        self.game = game
        self._game_to_prism_map = {n: i for i, n in enumerate(game._graph.nodes())}
        self._prism_to_game_map = dict(zip(self._game_to_prism_map.values(),
                                           self._game_to_prism_map.keys()))
        model_filename = self._export_prism_model()
        props_filename = self._export_prism_property()
        self._initialized = True

    def export_files_to_prism(self, game):
        self._initialize_variables(game)
        self._initialized = False

    def _export_prism_model(self) -> str:
        """
        Export TwoPlayerGame as a PRISM model to a file
        """
        graph_name = self.graph_name_in_prism

        # If filepath not given, then use graph_name as a filename
        directory = self._local_dir
        filepath = os.path.join(directory, graph_name + '.prism')

        # Create directory if not exists
        file_dir, _ = os.path.split(filepath)
        Path(file_dir).mkdir(parents=True, exist_ok=True)

        # Info about the graph
        players = self.game.players
        num_weight = len(self.game.weight_types)

        num_node = len(self.game._graph.nodes())
        if num_node < 2:
            warnings.warn('num_node should be greater or equal to 2')

        with open(filepath, 'w+') as f:
            # Stochastic Multi-Player Game
            f.write('smg\n\n')

            actions_per_player = defaultdict(lambda: set())
            for u_node_data in self.game._graph.nodes.data():
                u_node = u_node_data[0]
                player = u_node_data[1]['player']
                for v_node in self.game._graph.successors(u_node):
                    action = self.game.get_edge_attributes(u_node, v_node, 'actions')
                    if 'Init' in action:
                        action = 'Init'
                    if isinstance(action, set):
                        action = list(action)[0]
                        if isinstance(action, tuple):
                            action = '_'.join(action)
                    actions_per_player[player].add(action)

            for i, (player, actions) in enumerate(actions_per_player.items()):
                add_bracket = lambda s: f'[{s}]'
                action_list_str = ', '.join(map(add_bracket, actions))
                f.write(f'player p{i+1}\n')
                f.write(f'\t{action_list_str}\n')
                f.write(f'endplayer\n\n')

            # Game
            self._prism_action_idx_to_game_map = defaultdict(lambda: [])
            f.write(f'module {graph_name}\n')
            f.write(f'\tx : [0..{num_node-1}] init 0;\n') # TODO: What if num_node is 0 or 1?
            for edge in self.game._graph.edges.data():
                u_node_int = self._game_to_prism_map[edge[0]]
                v_node_int = self._game_to_prism_map[edge[1]]
                action = edge[2].get('actions')
                if 'Init' in action:
                    action = 'Init'
                if isinstance(action, set):
                    action = list(action)[0]
                    if isinstance(action, tuple):
                        action = '_'.join(action)
                # TODO: nondeterministic transitions
                f.write(f"\t[{action}] x={u_node_int} -> 1 : (x'={v_node_int});\n")
                self._prism_action_idx_to_game_map[u_node_int].append(action)
            f.write(f'endmodule\n\n')

            # Weight Objective
            for i_weight in range(num_weight):
                weight_name = self.game.weight_types[i_weight]
                f.write(f'rewards "{weight_name}"\n')
                for edge in self.game._graph.edges.data():
                    u_node_int = self._game_to_prism_map[edge[0]]
                    action = edge[2].get('actions')
                    if 'Init' in action:
                        action = 'Init'
                    if isinstance(action, set):
                        action = list(action)[0]
                        if isinstance(action, tuple):
                            action = '_'.join(action)
                    weight = edge[2].get('weights')[weight_name]
                    f.write(f'\t[{action}] x={u_node_int} : {weight};\n')
                f.write(f'endrewards\n\n')

            # Reachability Objective
            # Set weight of 1 to edges that lead to the accepting state
            # TODO: If there exists a few accepting states, then create a virtual node
            f.write(f'rewards "reach"\n')
            for v_node in self.game.get_accepting_states():
                for u_node in self.game._graph.predecessors(v_node):
                    if u_node == v_node:
                        continue
                    action = self.game.get_edge_attributes(u_node, v_node, 'actions')
                    if 'Init' in action:
                        action = 'Init'
                    if isinstance(action, set):
                        action = list(action)[0]
                        if isinstance(action, tuple):
                            action = '_'.join(action)
                    u_node_int = self._game_to_prism_map[u_node]
                    f.write(f'\t[{action}] x={u_node_int} : 1;\n')
            f.write(f'endrewards\n\n')

        return os.path.abspath(filepath)

    def _export_prism_property(self) -> str:
        """
        Export Multi-Objective Optimization property to a file
        """
        graph_name = self.graph_name_in_prism

        # If filepath not given, then use graph_name as a filename
        directory = self._local_dir
        filepath = os.path.join(directory, graph_name + '.props')

        # Create directory if not exists
        file_dir, _ = os.path.split(filepath)
        Path(file_dir).mkdir(parents=True, exist_ok=True)

        # TODO: Clean the following weight selection

        # Minimizes weights and maximize reachability
        with open(filepath, 'w+') as f:

            # if 'r' in list(self._max_weight_values.keys()):
            if self._max_weight_values is not None and 'r' in self._max_weight_values:
                max_weight_value = self._max_weight_values['r']
            else:
                max_weight_value = 0.98

            f.write(f'const double r = {max_weight_value};\n')

            for i, name in enumerate(self.game.weight_types):
                if self._max_weight_values is not None and name in self._max_weight_values:
                    max_weight_value = self._max_weight_values[name]
                else:
                    max_weight_value = self._get_max_weight_value(name)
                f.write(f'const double v{i} = {max_weight_value};\n')

            f.write('\n')
            weight_obj_str = [f'R{{"{name}"}}<=v{i}[C]' for i, name in enumerate(self.game.weight_types)]
            obj_str = ' & '.join([f'R{{"reach"}}>=r[C]'] + weight_obj_str)
            f.write(f'<<p1>> ({obj_str})\n\n')

        return os.path.abspath(filepath)

    def _get_max_weight_value(self, weight_name: str) -> float:
        """
        Compute the maximum possible value for the given weight
        """
        weights = [e[2]['weights'].get(weight_name) for e in self.game._graph.edges.data()]
        max_single_weight = max(weights)
        max_depth = len(self.game._graph.nodes())
        # successors = [list(self.game._graph.successors(n)) for n in self.game._graph.nodes()]
        # max_width = max([len(s) for s in successors])

        return max_single_weight * max_depth

    def _set_pareto_point(self, game, pareto_point):
        # TODO: Internally hold game
        game_ = copy.deepcopy(game)

        p_str = '_'.join(list(map(str, pareto_point[1:])))
        p_str = p_str.replace('.', '').replace(',','')
        game_.graph_name += f'_wo_loops_{p_str}'

        # p = [1.00, 5.50, 5.50]
        margin = [-0.01 * pareto_point[0]] + [0.01 * v for v in pareto_point[1:]]
        weight_names = ['r'] + self.weight_names
        max_weight_values = {k: v+m for k, v, m in zip(weight_names, pareto_point, margin)}
        self._set_max_weight_values(max_weight_values)

        return game_

    def _set_max_weight_values(self, weights: Dict) -> None:
        weight_names = list(weights.keys())
        if weight_names != ['r']+self.weight_names:
            raise ValueError(f'Invalid weight names. The keys should be {self.weight_names}')

        for w in weights.values():
            if not isinstance(w, (int, float)):
                raise ValueError(f'Invalid weight values. The values should be int or float')

        self._max_weight_values = weights

    def _create_prism_filenames(self, extensions: List, directory: str, filename: str) -> Dict:
        """
        Simply create several filenames for different extensions

        :args keys:     Just keys
        """
        filenames = {}
        for ext in extensions:
            path = os.path.join(directory, filename + ext)
            filenames[ext] = path
        return filenames

    def _plot_pareto(self, pareto_points: List):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for p in pareto_points:
            ax.scatter(p[1], p[2], p[0])
            ax.set_xlabel('Game Cost')
            ax.set_ylabel('Preference Cost')
            ax.set_zlabel('Reachability')
        plt.show()

    def _find_pareto_points_from_stdout(self, completed_process: str,
                                        only_reachable: bool = True) -> List:
        """
        Find pareto points from PRISM std output
        Example of an output:
        > maxcorners=6. state 0:
        > [[-0.0000, 10.0000, 9.2104]r:[-1.0000, 0.0000, 0.0000]r:[0.0000, 0.0000, 1.0000]r:[0.0000, 1.0000, 0.0000][1.0000, 10.0000, 14.1440][1.0000, 15.0000, 9.5389]]

        One brutal way to find the pareto points is to find a word 'maxcorners'
        and get the next line.
        """
        if not isinstance(completed_process, str):
            return

        pareto_points = None
        found_maxcorners = False

        for line in completed_process.split('\n'):
            if found_maxcorners:
                # Find all that is in brackets
                pareto_points = re.findall(r'\[.*?\]', line)
                # Find all floats in a bracket
                pareto_points = [re.findall(f'[-+]?\d*\.\d+|\d+', p) for p in pareto_points]
                pareto_points = [[float(f) for f in p] for p in pareto_points]
                found_maxcorners = False

            if 'maxcorners' in line:
               found_maxcorners = True

        if only_reachable:
            reachability_index = 0
            pareto_points = [p for p in pareto_points if p[reachability_index] > 0]

        return pareto_points

    def _read_strategy(self, sta_path: str, adv_path: str) -> Dict:
        """
        Read adv file and extract adversaries (strategy)

        :args   sta_path:           .sta file path
        :args   adv_path:           .adv file path

        :return game_strategy:      A strategy on TwoPlayerGame
        """
        if not self._initialized:
            raise Exception('Not Initialize. Please first run run_prism')

        if not os.path.exists(sta_path):
            warnings.warn(f'There is not file called {sta_path}')
            return
        if not os.path.exists(adv_path):
            warnings.warn(f'There is not file called {adv_path}')
            return

        # Read .sta file

        # A template for finding '(STA_STATE): (MODEL_STATE) ...'
        template = r'(\d+)\:\((\d+)\)\n'
        # Read line by line
        try:
            self._sta_to_prism_state_map = {}
            with open(sta_path) as f:
                line = f.readline()
                while line:
                    m = re.match(template, line)
                    if m is None:
                        line = f.readline()
                        continue
                    sta_state = int(m.group(1))
                    prism_state = int(m.group(2))
                    self._sta_to_prism_state_map[sta_state] = prism_state
                    line = f.readline()
        except:
            raise Exception("Could not read .sta file. Each line of .sta file should be in the" + \
                            "form of '(STA_STATE): (MODEL_STATE) ...'")

        # Read .adv file

        # There are two types of adv file: Simple and Complicated (idk why)
        # CLI outputs the simple one and GUI outputs the complicated one
        try:
            with open(adv_path) as f:
                line = f.readline()
                # Very Hacky Way to check whether it's simple/complicated
                if '$' in line:
                    game_strategy = self._read_complicated_adv(f, line)
                    # game_strategy = self._convert_strategy(prism_strategy)
                else:
                    # Simple one doesn't require .sta file
                    game_strategy = self._read_simple_adv(f, line)
        except:
            raise Exception("Could not read .adv file.")

        return game_strategy

    def _read_complicated_adv(self, f, line) -> Dict:
        """
        Read .adv file and extract the strategy that prism-games computed.
        It looks like there are 3 strategies in the file.
        They all look the same, but strategy under 'MemUpdMoves' looks more
        informative than other two strategies.

        We read line by line until we hit a word `Info: ...`.

        Then, we convert the strategy into the strategy on TwoPlayerGame,
        so that we can use in this library

        :args   f:      File
        :args line:     Each line in file
        """
        while 'InitState:' not in line:
            line = f.readline()

        # Get the initial state
        initial_state = None
        while 'Init:' not in line:
            m = re.match(r'(\d+)', line)
            if m is None:
                line = f.readline()
                continue
            initial_state = int(m.group(1))
            line = f.readline()

        init_dist: DistOverPareto = defaultdict(lambda: None)
        update_state_dict: Dict[Node, Dict[ParetoPoint, Dict[Action, DistOverPareto]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None))))
        update_move_dict: Dict[Node, Dict[Action, Dict[ParetoPoint, Node]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))

        # Get the initial distribution
        while 'Next:' not in line:
            corners = re.findall(r'(\d+)\=', line)
            probs = re.findall(r'\=(\d+\.\d+)', line)
            if len(corners) == len(probs) == 0:
                line = f.readline()
                continue
            corners = list(map(int, corners))
            probs = list(map(float, probs))
            probs = [v / sum(probs) for v in probs]

            init_dist = {corner: prob for corner, prob in zip(corners, probs)}
            line = f.readline()

        # Get the Next Function: S x M -> A
        while 'MemUpdStates:' not in line:
            m = re.match(r'(\d+)\s(\d+)\s', line)
            if m is None:
                line = f.readline()
                continue

            curr_state = int(m.group(1))
            curr_corner = int(m.group(2))

            action_indices = re.findall(r'(\d+)\=', line)
            probs = re.findall(r'\=(\d+\.\d+)', line)
            action_indices = list(map(int, action_indices))
            probs = list(map(float, probs))
            probs = [v / sum(probs) for v in probs]

            line = f.readline()

        # Get the Memory Update States Function: S x M x A -> D(M)
        while 'MemUpdMoves:' not in line:
            m = re.match(r'(\d+)\s(\d+)\s(\d+)\s', line)
            if m is None:
                line = f.readline()
                continue

            next_corners = re.findall(r'(\d+)\=', line)
            probs = re.findall(r'\=(\d+\.\d+)', line)
            next_corners = list(map(int, next_corners))
            probs = list(map(float, probs))
            probs = [v / sum(probs) for v in probs]

            state_idx = int(m.group(1))
            curr_corner = int(m.group(2))
            action_idx = int(m.group(3))

            state_in_prism = self._sta_to_prism_state_map[state_idx]
            state = self._prism_to_game_map[state_in_prism]
            action = self._prism_action_idx_to_game_map[state_in_prism][action_idx]
            for next_corner, prob in zip(next_corners, probs):
                update_state_dict[state][curr_corner][action][next_corner] = prob

            line = f.readline()

        # Get the Memory Update Moves Function: S x A x M -> S x D(M)
        while 'Info:' not in line:
            m = re.match(r'(\d+)\s(\d+)\s(\d+)\s(\d+)\s', line)
            if m is None:
                line = f.readline()
                continue

            states = re.findall(r'(\d+)\=', line)
            probs = re.findall(r'\=(\d+\.\d+)', line)
            states = list(map(int, states))
            probs = list(map(float, probs))
            probs = [v / sum(probs) for v in probs]

            # (CURR_NODE) (ACTION_IDX) (CORNER?) (NEXT_NODE)
            u_node_in_adv = int(m.group(1))
            action_idx = int(m.group(2))
            curr_corner = int(m.group(3))
            v_node_in_adv = int(m.group(4))

            u_node_in_prism = self._sta_to_prism_state_map[u_node_in_adv]
            v_node_in_prism = self._sta_to_prism_state_map[v_node_in_adv]
            u_node = self._prism_to_game_map[u_node_in_prism]
            v_node = self._prism_to_game_map[v_node_in_prism]
            action = self._prism_action_idx_to_game_map[u_node_in_prism][action_idx]

            update_move_dict[u_node][action][curr_corner] = v_node
            line = f.readline()

        strategy = StochasticStrategy(
            self.game,
            update_state_dict,
            update_move_dict,
            init_dist=init_dist,
            graph_name=f'StochasticStrategyFor{self.graph_name_in_prism}')
        strategy.plot_graph()
        strategy.plot_original_graph()

        return strategy

    def _read_simple_adv(self, f, line) -> Dict:
        """
        Read .adv file and extract the strategy that prism-games computed.
        Each line includes (CURR_NODE) (ACTION). That's it!

        :args   f:      File
        :args line:     Each line in file
        """
        game_strategy = defaultdict()
        # A template to find "(CURR_NODE) (ACTION)"
        template = r'(\d+)\s(\S+)'

        while line:
            if line == '\n':
                break
            m = re.match(template, line)
            u_node_in_prism_model = int(m.group(1))
            action_name = m.group(2)
            u_node_in_game = self._prism_to_game_map[u_node_in_prism_model]
            for v_node_in_game in self.game._graph.successors(u_node_in_game):
                a = self.game.get_edge_attributes(u_node_in_game, v_node_in_game, 'actions')
                if a == action_name:
                    game_strategy[u_node_in_game] = [{'next_node': v_node_in_game, 'action': action_name}]
            line = f.readline()
        return game_strategy

    def delete_loops(self, game):
        if game is None:
            return None

        game_wo_loops = copy.deepcopy(solver._game)

        edges_to_delete = set()
        cycles = game_wo_loops.get_cycles()
        node_to_cycles = defaultdict(lambda: [])

        for i, cycle in enumerate(cycles):
            for node in cycle:
                node_to_cycles[node].append(i)

        for cycle in cycles:
            if 'Accepting' in cycle:
                continue

            for u_node in cycle:
                player = game_wo_loops.get_state_w_attribute(u_node, 'player')
                if player == 'adam':
                    continue

        for edge in edges_to_delete:
            game_wo_loops._graph.remove_edge(*edge)

        game_wo_loops._sanity_check()
        game_wo_loops.plot_graph()

        return game_wo_loops

    @property
    def graph_name_in_prism(self):
        """
        Transform graph_name into the prism form
        """
        return self.game._graph_name.replace(' ', '')

    @property
    def strategy(self) -> Dict:
        if self._game_str is None:
            warnings.warn('Please first run run_prism')

        return self._game_str

    @property
    def weight_names(self) -> List:
        return self.game.weight_types

    @property
    def weight(self, plan: List, weight_name: str) -> float:
        curr_state = self.game.get_initial_states()[0][0]
        accp_state = self.game.get_accepting_states()[0]

        sum_weight = 0.0

        for action in plan:
            for edge in self.game._graph.successors(curr_state):
                if edge[2]['action'] == action:
                    weight = edge[2]['weights'][weight_name]
                    sum_weight += weight
                    curr_state = edge[1]

        if curr_state != accp_state:
            warnings.warn('Did not end at an accepting state')

        return sum_weight

    @property
    def pareto_points(self):
        if self._pareto_points is None:
            warnings.warn('Run run_prism first')
        return self._pareto_points

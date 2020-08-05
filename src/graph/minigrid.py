import warnings
import sys

from typing import Tuple, Optional, Iterable
from graphviz import Digraph

# import local packages
from src.graph import FiniteTransSys
from src.factory.builder import Builder


class MiniGrid(FiniteTransSys):

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False):
        FiniteTransSys.__init__(self, graph_name, config_yaml, save_flag)

    def build_graph_from_file(self):
        """
        A method to build the graph from a config file. Before we run this method we need to make sure that the
        graph data like nodes and edges and their respective attributes have been store in a self._graph_yaml attribute.
        :return: updates the graph with the respective nodes and the edges
        """

        if self._graph_yaml is None:
            warnings.warn("Please ensure that you have first loaded the config data. You can do this by"
                          "setting the respective True in the builder instance.")

        _nodes = self._graph_yaml['nodes']
        _start_state = self._graph_yaml['start_state']

        # each node has an atomic proposition and a player associated with it. Some states also init and
        # accepting attributes associated with them
        for _n in _nodes:
            state_name = _n[0]
            ap = _n[1].get('observation')
            # all nodes we get from the gym minigrid be default belong to system/eve
            # player = _n[1].get('player')
            self.add_state(state_name, ap=ap, player="eve")

            if _n[1].get('is_accepting'):
                self.add_accepting_state(state_name)

        self.add_initial_state(_start_state)

        _edges = self._graph_yaml['edges']

        # as originally the minigrid world does not have weight associated with its actions,
        # we will manually assign a weight here

        ACTION_STR_TO_WT = {
            'north': -1,
            'south': -1,
            'east': -1,
            'west': -1,
            'northeast': -1,
            'northwest': -1,
            'southeast': -1,
            'southwest': -1
        }

        # NOTE : ALL actions have the same cost of 1 unless specified in the yaml specifically
        for _e in _edges:
            _weight = _e[2].get('weight')
            _action = _e[2].get('label')

            if _weight is None:
                self.add_edge(_e[0], _e[1], weight=ACTION_STR_TO_WT[_action], actions=_action)
            else:
                self.add_edge(_e[0], _e[1], weight=_weight, actions=_action)

    def from_raw_minigrid_TS(self,
                         human_interventions: int = 1,
                         plot_raw_ts: bool = False,
                         debug: bool = False) -> 'MiniGrid()':
        self._sanity_check(debug=debug)
        minigrid_game_ts = self.automate_construction(human_interventions, plot_raw_ts, debug=debug)
        self._graph = minigrid_game_ts._graph

        return minigrid_game_ts

    # a method that builds a deterministic TS with human nodes from a raw abstraction from gym-minigrid env
    # with sys nodes only
    def automate_construction(self, k: int, plot_raw_ts: bool = False, debug: bool = False) -> FiniteTransSys:
        """
        Given a TS with only the system node, we add human nodes after every transition that the system can take.
        The human can pick the robot in a 2d grid world and place it in any of its neighbouring cells. He/she can
        intervene only k times.

        :param k: # of time the human can intervene
        :return:
        """
        if plot_raw_ts:
            self.fancy_graph()

        eve_node_lst = []
        adam_node_lst = []
        two_player_graph_ts = FiniteTransSys(f"game_{self._graph_name}", f"config/minigrid_game_TS", self._save_flag)
        two_player_graph_ts.construct_graph()

        # lets create k copies of the states
        for _n in self._graph.nodes():
            for i in range(k + 1):

                _x, _y = self.__get_pos_from_minigrid_state(_n)

                _sys_node = ((_x, _y), i)

                if _sys_node in eve_node_lst:
                    warnings.warn(f"The graph contains multiple states with the same position."
                                  f"Please make sure your abstraction is based on a directionless agent form wombats."
                                  f"The state that was repeated more than once is {_sys_node}")
                else:
                    eve_node_lst.append(_sys_node)

        self._add_game_states_from(two_player_graph_ts, eve_node_lst, player="eve")

        # for each edge create a human node and then alter the original edge to go through the human node
        for _e in self._graph.edges():
            for i in range(k + 1):
                _u = _e[0]
                _v = _e[1]

                _ux, _uy = self.__get_pos_from_minigrid_state(_u)
                _vx, _vy = self.__get_pos_from_minigrid_state(_v)

                # lets create a human edge with huv,k naming convention
                _env_node = ((f"h{(_ux, _uy)}{(_vx, _vy)}"), i)
                adam_node_lst.append(_env_node)

        self._add_game_states_from(two_player_graph_ts, adam_node_lst, player="adam")

        # add init node
        _init_node = self.get_initial_states()
        _ix, _iy = self.__get_pos_from_minigrid_state(_init_node[0][0])
        two_player_graph_ts.add_initial_state(((_ix, _iy), k))

        self._build_game_edges(human_interventions=k, two_player_game=two_player_graph_ts)

        # add the original atomic proposition to the new states
        for _n in self._graph.nodes.data():
            if _n[1].get('ap'):
                _x, _y = self.__get_pos_from_minigrid_state(_n[0])

                for ik in range(k + 1):
                    two_player_graph_ts.add_state_attribute(((_x, _y), ik), 'ap', _n[1].get('ap'))

        return two_player_graph_ts

    def _build_game_edges(self, human_interventions: int, two_player_game: FiniteTransSys):
        for _e in self._graph.edges.data():
            _u = _e[0]
            _v = _e[1]
            _attr = _e[2]

            _ux, _uy = self.__get_pos_from_minigrid_state(_u)
            _vx, _vy = self.__get_pos_from_minigrid_state(_v)

            for ik in reversed(range(human_interventions + 1)):
                if ik != 0:
                    self._build_game_transitions_ik(two_player_game, _ux, _uy, _vx, _vy, _attr, ik)
                else:
                    self._build_game_transition(two_player_game, _ux, _uy, _vx, _vy, _attr)

    def _build_game_transition(self,
                               _game: FiniteTransSys,
                               _ux: int,
                               _uy: int,
                               _vx: int,
                               _vy: int,
                               _attr: dict,):
        # add edge from sys node to human node
        self._add_game_transition(_game,
                                  _u_game_state=((_ux, _uy), 0),
                                  _v_game_state=((f"h{(_ux, _uy)}{(_vx, _vy)}"), 0),
                                  actions=_attr.get("actions"),
                                  weight=_attr.get("weight"))

        # if the human decides not to take any action then we proceed as per the original transition
        self._add_game_transition(_game,
                                  _u_game_state=((f"h{(_ux, _uy)}{(_vx, _vy)}"), 0),
                                  _v_game_state=((_vx, _vy), 0),
                                  actions=_attr.get("actions"),
                                  weight=0)


    def _build_game_transitions_ik(self,
                                _game: FiniteTransSys,
                                _ux: int,
                                _uy: int,
                                _vx: int,
                                _vy: int,
                                _attr: dict,
                                ik: int):
        # add edge from sys node to human node
        self._add_game_transition(_game,
                                  _u_game_state=((_ux, _uy), ik),
                                  _v_game_state=((f"h{(_ux, _uy)}{(_vx, _vy)}"), ik),
                                  actions=_attr.get("actions"),
                                  weight=_attr.get("weight"))

        # if the human decides not to take any action then we proceed as per the original transition
        self._add_game_transition(_game,
                                  _u_game_state=((f"h{(_ux, _uy)}{(_vx, _vy)}"), ik),
                                  _v_game_state=((_vx, _vy), ik),
                                  actions=_attr.get("actions"),
                                  weight=0)

        # add transition from a human node to the neighbouring nodes
        # check if _ux + 1 and _ux - 1 exists. Similarly, check if _uy + 1 and _uy - 1 exists
        cells = self.__get_neighbouring_cells(_ux, _uy, get_current_cell=True)

        for cell in cells:
            # if cell != (_vx, _vy):
            if _game._graph.has_node((cell, ik - 1)):
                self._add_game_transition(_game,
                                          _u_game_state=((f"h{(_ux, _uy)}{(_vx, _vy)}"), ik),
                                          _v_game_state=(cell, ik - 1),
                                          actions="m",
                                          weight=0)

    def _add_game_transition(self,
                             _game: FiniteTransSys,
                             _u_game_state: tuple,
                             _v_game_state: tuple,
                             **game_edge_attr) -> None:
        """
        A helper method to add an edge to the augmented game if it already does not exists given the current game state
        and the next game state we can transit to.

        :return:
        """

        if not _game._graph.has_edge(_u_game_state, _v_game_state):
            _game.add_edge(_u_game_state, _v_game_state, **game_edge_attr)

    def _add_game_state(self,
                        _game: FiniteTransSys,
                        _game_state: tuple,
                        **game_node_attr) -> None:
        """
        A helper method to add a node to the given game instabce if it already does not exists.
        :param _game_state:
        :return:
        """

        if not _game._graph.has_node(_game_state):
            _game.add_state(_game_state, **game_node_attr)

    def _add_game_states_from(self,
                              _game: FiniteTransSys,
                              _game_states: Iterable,
                              **game_node_attr) -> None:
        """
        A helper method to add a node to the given game instabce if it already does not exists.
        :param _game_state:
        :return:
        """

        for _game_state in _game_states:
            self._add_game_state(_game, _game_state, **game_node_attr)

    def fancy_graph(self, color=()) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["nodes"]
        for n in nodes:
            ap = n[1].get('xlabel')
            if ap is None:
                ap = n[1].get('ap')
            color = n[1].get('color')

            if not isinstance(n[0], str):
                _node_name = str(n[0])
            else:
                _node_name = n[0]
            dot.node(_node_name, _attributes={"style": "filled", "fillcolor": color, "xlabel": ap, "shape": "rectangle"})
            # if n[1].get("player") == "adam":
            #     dot.node(_node_name, _attributes={"shape": "circle"})

        # add all the edges
        edges = self._graph_yaml["edges"]

        # load the weights to illustrate on the graph
        for counter, edge in enumerate(edges):
            label = edge[2].get('label')
            if label is None:
                label = str(edge[2].get('weight'))
            fontcolor = edge[2].get('fontcolor')
            dot.edge(str(edge[0]), str(edge[1]), label=label, fontcolor=fontcolor)

        # set graph attributes
        # dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            graph_name = str(self._graph.__getattribute__('name'))
            self.save_dot_graph(dot, graph_name, True)

    def __get_neighbouring_cells(self, _ux, _uy, get_current_cell: bool = False) -> Tuple:
        """
        A helper method that returns set of neighbouring cells given x, y position of a cell.

        :param _ux:
        :param _uy:
        :param _get_current_cell: if true, then return the current position (x, y) else dont
        :return:
        """

        if not isinstance(_ux, int) or not isinstance(_uy, int):
            warnings.warn(f"Please make sure that the positions of the cells are integers."
                          f"Currently there are of type {(type(_ux), type(_uy))} for {(_ux, _uy)}")

        _east_cell = (_ux, _uy + 1)
        _west_cell = (_ux, _uy - 1)
        _north_cell = (_ux + 1, _uy)
        _south_cell = (_ux - 1, _uy)

        if get_current_cell:
            return _east_cell, _west_cell, _north_cell, _south_cell, (_ux, _uy)
        else:
            return _east_cell, _west_cell, _north_cell, _south_cell

    def __get_pos_from_minigrid_state(self, _state: str) -> Tuple[int, int ]:
        """
        A helper method that extract the x and y position of a given node.

        Given a node of the form : (1,1), right
        :return:
        """

        x, y = _state.split("(")[1].split(")")[0].split(",")

        try:
            return int(x), int(y)
        except ValueError:
            print(f"Error converting the position of a cell to int."
                  f" The position of the cell is not an integer."
                  f"(x, y) = {x, y}. These value are directly abstracted from wombats abstraction."
                  f"Possible sources of error could be readings the file or the dumping into the abstraction.")


class MiniGridBuilder(Builder):

    def __init__(self):
        Builder.__init__(self)

    def __call__(self,
                 graph_name: str,
                 config_yaml: str,
                 human_intervention: int = 1,
                 raw_minigrid_ts: Optional[MiniGrid] = None,
                 save_flag: bool = False,
                 plot: bool = False,
                 plot_raw_minigrid: bool = False,
                 debug: bool = False) -> 'MiniGrid()':

        """
        A function to build the TS from the gym-minigrid env from a config_file.

        NOTE: For now we only have the provision to build this TS from a yaml file
        :param graph_name:
        :param config_yaml:
        :param save_flag:
        :param plot:
        :return:
        """
        if not isinstance(human_intervention, int):
            try:
                human_intervention = int(human_intervention)
            except ValueError:
                warnings.warn("Please make sure the number of times the human can intervene is integer. e.f 1, 1.0.")
                sys.exit(-1)

        if raw_minigrid_ts:
            self._instance = raw_minigrid_ts.from_raw_minigrid_TS(human_interventions=human_intervention,
                                                 plot_raw_ts=plot_raw_minigrid,
                                                 debug=debug)
        else:
            self._instance = MiniGrid(graph_name, config_yaml, save_flag=save_flag)
            self._instance.construct_graph()
            self._instance._graph_yaml = self._from_yaml(config_yaml)
            self._instance.build_graph_from_file()

        if plot:
            self._instance.plot_graph()

        return self._instance

    def _from_yaml(self, config_file_name: str) -> dict:

        config_data = self.load_YAML_config_data(config_file_name)

        return config_data
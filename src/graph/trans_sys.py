import warnings
import math

from graphviz import Digraph
from typing import List, Tuple, Dict, Optional

# local packages
from .two_player_graph import TwoPlayerGraph
from ..factory.builder import Builder


class FiniteTransSys(TwoPlayerGraph):

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False):
        TwoPlayerGraph.__init__(self, graph_name, config_yaml, save_flag)

    def fancy_graph(self, color=("lightgrey", "red", "purple")) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["nodes"]
        for n in nodes:
            # default color for all the nodes is grey
            ap = n[1].get('ap')
            ap = "{" + str(ap) + "}"
            dot.node(str(n[0]), _attributes={"style": "filled",
                                             "fillcolor": color[0],
                                             "xlabel": ap,
                                             "shape": "rectangle"})
            if n[1].get('init'):
                # default color for init node is red
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[1], "xlabel": ap})
            if n[1].get('accepting'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[2], "xlabel": ap})
            if n[1].get('player') == 'eve':
                dot.node(str(n[0]), _attributes={"shape": "rectangle", "xlabel": ap})
            if n[1].get('player') == 'adam':
                dot.node(str(n[0]), _attributes={"shape": "circle", "xlabel": ap})

        # add all the edges
        edges = self._graph_yaml["edges"]

        # load the weights to illustrate on the graph
        for counter, edge in enumerate(edges):
            if edge[2].get('strategy') is True:
                dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('actions')),
                         _attributes={'color': 'red'})
            else:
                dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('actions')))

        # set graph attributes
        # dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            graph_name = str(self._graph.__getattribute__('name'))
            self.save_dot_graph(dot, graph_name, True)

    # a function to construct the two game automatically in code. K = # of times the human can intervene
    def automate_construction(self, k: int):
        if not isinstance(k, int):
            warnings.warn("Please Make sure the Quantity K which represents the number of times the human can "
                          "intervene is an integer")
        eve_node_lst = []
        adam_node_lst = []
        two_player_graph_ts = FiniteTransSys(self._graph_name, self._config_yaml, self._save_flag)
        two_player_graph_ts.construct_graph()

        # lets create k copies of the states
        for _n in self._graph.nodes():
            for i in range(k+1):
                _sys_node = (_n, i)
                eve_node_lst.append(_sys_node)

        two_player_graph_ts.add_states_from(eve_node_lst, player='eve')

        # for each edge create a human node and then alter the original edge to go through the human node
        for e in self._graph.edges():
            for i in range(k):
                # lets create a human edge with huv,k naming convention
                _env_node = ((f"h{e[0][1:]}{e[1][1:]}"), f"{i}")
                adam_node_lst.append(_env_node)

        two_player_graph_ts.add_states_from(adam_node_lst, player='adam')

        # add init node
        init_node = self.get_initial_states()
        two_player_graph_ts.add_state_attribute((init_node[0][0], 0), "init", True)

        for e in self._graph.edges.data():
            # add edge between e[0] and the human node h{e[0][1:]}{e[1][1:]}, k
            for ik in range(k):
                two_player_graph_ts.add_edge((e[0], ik), ((f"h{e[0][1:]}{e[1][1:]}"), f"{ik}"),
                                             actions=e[2].get("actions"), weight=e[2].get("weight"))
                two_player_graph_ts.add_edge(((f"h{e[0][1:]}{e[1][1:]}"), f"{ik}"), (e[1], ik),
                                             actions=e[2].get("actions"), weight=e[2].get("weight"))

                _alt_nodes_set = set(self._graph.nodes()) - {e[1]}
                for _alt_node in _alt_nodes_set:
                    two_player_graph_ts.add_edge(((f"h{e[0][1:]}{e[1][1:]}"), f"{ik}"), (_alt_node, ik+1),
                                                 actions="m", weight=0)

        # manually add edges to states that belong to k index
        for e in self._graph.edges.data():
            two_player_graph_ts.add_edge((e[0], k), (e[1], k),
                                         actions=e[2].get('actions'), weight=e[2].get("weight"))

        # add the original atomic proposition to the new states
        for _n in self._graph.nodes.data():
            if _n[1].get('ap'):
                for ik in range(k+1):
                    two_player_graph_ts.add_state_attribute((_n[0], ik), 'ap', _n[1].get('ap'))

        return two_player_graph_ts

    def _sanity_check(self, debug: bool = False):
        """
        A helper method that loops through every node and checks if it has an outgoing edge or not.
        If not then we add action "self" with weight 0 if its an accepting states else -1 * |max_weight|.
        :return:
        """
        max_weight = self.get_max_weight()
        accn_states = self.get_accepting_states()
        for _n in self._graph.nodes():
            if len(list(self._graph.successors(_n))) == 0:
                if debug:
                    print("====================================")
                    print(f"Adding a self loop to state {_n} in {self._graph.name}")
                    print("====================================")
                # if its an accepting state
                if _n in accn_states:
                    self._graph.add_edge(_n, _n, weight=0, actions="self")
                # if its a trap state
                else:
                    self._graph.add_edge(_n, _n, weight=max_weight, actions="self")

    def _sanity_check_finite(self, debug: bool = False):
        """
        A helper method that loops through every node and checks if it has an outgoing edge or not.
        If not then we add action "self" with weight {}.
        :return:
        """
        max_weight = -1 * math.inf
        accn_states = self.get_accepting_states()
        for _n in self._graph.nodes():
            if len(list(self._graph.successors(_n))) == 0:
                if debug:
                    print("====================================")
                    print(f"Adding a self loop to state {_n} in {self._graph.name}")
                    print("====================================")
                # if its an accepting state
                if _n in accn_states:
                    self._graph.add_edge(_n, _n, weight=0, actions="self")
                # if its a trap state
                else:
                    self._graph.add_edge(_n, _n, weight=max_weight, actions="self")

    @classmethod
    def from_raw_ts(cls, raw_ts: TwoPlayerGraph,
                    graph_name: str,
                    config_yaml: str,
                    save_flag: bool = False,
                    plot: bool = False,
                    human_intervention: int = 1,
                    plot_raw_ts: bool = False,
                    finite: bool = False,
                    debug: bool = False):
        """
        Return a concrete instance of a FiniteTransSys given a basic transition system with nodes
        that belong only to the system(eve)
        :param raw_ts:
        :param graph_name:
        :param config_yaml:
        :param save_flag:
        :param plot:
        :param human_intervention:
        :param plot_raw_ts:
        :param debug:
        :return: An instance of the FiniteTransSys that contains both the env(adam) and sys(eve) nodes
        """
        raw_trans_name = "raw" + graph_name
        trans_sys = FiniteTransSys(raw_trans_name, f"config/{raw_trans_name}", save_flag=save_flag)
        trans_sys._graph = raw_ts._graph
        if finite:
            trans_sys._sanity_check_finite(debug=debug)
        else:
            trans_sys._sanity_check(debug=debug)

        if plot_raw_ts:
            trans_sys.plot_graph()

        trans_sys._graph_name = graph_name
        trans_sys._config_yaml = config_yaml
        trans_sys = trans_sys.automate_construction(k=human_intervention)

        if plot:
            trans_sys.plot_graph()

        if debug:
            trans_sys.print_nodes()
            trans_sys.print_edges()

        return trans_sys

    @classmethod
    def get_three_state_ts(cls, graph_name: str,
                           config_yaml: str,
                           save_flag: bool = False,
                           debug: bool = False,
                           plot: bool = False,
                           human_intervention: int = 1,
                           plot_raw_ts: bool = False):
        """
        A methods that return a concrete instance of FiniteTransitionSystem with eve and adam nodes and edges.
        :param graph_name:
        :param config_yaml:
        :param save_flag:
        :param debug:
        :param plot:
        :param human_intervention:
        :param plot_raw_ts:
        :return:
        """

        raw_trans_name = "raw" + graph_name
        trans_sys = FiniteTransSys(raw_trans_name, f"config/{raw_trans_name}", save_flag=save_flag)
        trans_sys.construct_graph()

        trans_sys.add_states_from(['s1', 's2', 's3'])
        trans_sys.add_state_attribute('s1', 'ap', 'b')
        trans_sys.add_state_attribute('s2', 'ap', 'a')
        trans_sys.add_state_attribute('s3', 'ap', 'c')
        trans_sys.add_edge('s1', 's2', actions='s12', weight=-0)
        trans_sys.add_edge('s2', 's1', actions='s21', weight=-2)
        trans_sys.add_edge('s2', 's3', actions='s23', weight=-3)
        trans_sys.add_edge('s3', 's1', actions='s31', weight=-5)
        trans_sys.add_edge('s1', 's3', actions='s13', weight=-3)
        # trans_sys.add_edge('s1', 's2', actions='s12', weight=-1)
        # trans_sys.add_edge('s2', 's1', actions='s21', weight=-1)
        # trans_sys.add_edge('s2', 's3', actions='s23', weight=-1)
        # trans_sys.add_edge('s3', 's1', actions='s31', weight=-1)
        # trans_sys.add_edge('s1', 's3', actions='s13', weight=-1)

        trans_sys.add_initial_state('s2')

        if plot_raw_ts:
            trans_sys.plot_graph()

        trans_sys._graph_name = graph_name
        trans_sys._config_yaml = config_yaml
        trans_sys = trans_sys.automate_construction(k=human_intervention)

        if plot:
            trans_sys.plot_graph()

        if debug:
            trans_sys.print_nodes()
            trans_sys.print_edges()

        return trans_sys

    @classmethod
    def get_five_state_ts(cls, graph_name: str,
                           config_yaml: str,
                           save_flag: bool = False,
                           debug: bool = False,
                           plot: bool = False,
                           human_intervention: int = 1,
                           plot_raw_ts: bool = False):
        """
        A methods that return a concrete instance of FiniteTransitionSystem with eve and adam nodes and edges.
        :param graph_name:
        :param config_yaml:
        :param save_flag:
        :param debug:
        :param plot:
        :param human_intervention:
        :param plot_raw_ts:
        :return:
        """

        raw_trans_name = "raw" + graph_name
        trans_sys = FiniteTransSys(raw_trans_name, f"config/{raw_trans_name}", save_flag=save_flag)
        trans_sys.construct_graph()

        trans_sys.add_states_from(['s1', 's2', 's3', 's4', 's5'])
        trans_sys.add_state_attribute('s1', 'ap', 'b')
        trans_sys.add_state_attribute('s2', 'ap', 'i')
        trans_sys.add_state_attribute('s3', 'ap', 'r')
        trans_sys.add_state_attribute('s4', 'ap', 'g')
        trans_sys.add_state_attribute('s5', 'ap', 'd')
        # E = 4 ; W = 2; S = 3 ; N = 9
        trans_sys.add_edge('s1', 's2', actions='E', weight=-4)
        trans_sys.add_edge('s2', 's1', actions='W', weight=-2)
        trans_sys.add_edge('s3', 's2', actions='N', weight=-9)
        trans_sys.add_edge('s2', 's3', actions='S', weight=-3)
        trans_sys.add_edge('s3', 's4', actions='S', weight=-3)
        trans_sys.add_edge('s4', 's3', actions='N', weight=-9)
        trans_sys.add_edge('s1', 's4', actions='W', weight=-2)
        trans_sys.add_edge('s4', 's1', actions='W', weight=-2)
        trans_sys.add_edge('s4', 's5', actions='E', weight=-4)
        trans_sys.add_edge('s5', 's4', actions='S', weight=-3)
        trans_sys.add_edge('s2', 's5', actions='E', weight=-4)
        trans_sys.add_edge('s5', 's2', actions='N', weight=-9)

        trans_sys.add_initial_state('s1')

        if plot_raw_ts:
            trans_sys.plot_graph()

        trans_sys._graph_name = graph_name
        trans_sys._config_yaml = config_yaml
        trans_sys = trans_sys.automate_construction(k=human_intervention)

        if plot:
            trans_sys.plot_graph()

        if debug:
            trans_sys.print_nodes()
            trans_sys.print_edges()

        return trans_sys


class TransitionSystemBuilder(Builder):

    def __init__(self):

        # call the parent class constructor
        Builder.__init__(self)
        self._pre_built = {}

    def __call__(self,
                 raw_trans_sys: FiniteTransSys,
                 graph_name: str,
                 config_yaml: str,
                 from_file: bool = False,
                 pre_built: bool = True,
                 built_in_ts_name: str = "",
                 save_flag: bool = False,
                 debug: bool = False,
                 plot: bool = False,
                 human_intervention: int = 1,
                 finite: bool = False,
                 plot_raw_ts: bool = False) -> 'FiniteTransSys':
        """
        A method to create an instance of a finite transition system consisting of two players - eve and system .
        :param raw_trans_sys: The original graph with only nodes that belong to eve.
        :param debug:
        :param plot:
        :param human_intervention:
        :param plot_raw_ts:
        :return:
        """

        print(f"No. of times the human can intervene is : {human_intervention}")

        if pre_built and built_in_ts_name == "":
            raise TypeError("Using the built in transition system. enter a valid transition system name.")

        self._instance = FiniteTransSys(graph_name, config_yaml, save_flag=save_flag)
        self._instance.construct_graph()

        # load dict with function calls
        self._load_pre_built()

        if pre_built:
            self._instance = self._from_built_in_ts(built_in_ts_name,
                                                    graph_name,
                                                    config_yaml,
                                                    save_flag,
                                                    debug,
                                                    plot,
                                                    human_intervention,
                                                    plot_raw_ts)

        elif raw_trans_sys:
            if not isinstance(raw_trans_sys, FiniteTransSys):
                raise TypeError(f"Please ensure that the raw transition system is of type {FiniteTransSys.__name__}. \n"
                                f"If you are trying to constructing a two player graph with sys(eve) and env(adam) nodes"
                                f" then use the builder for the {TwoPlayerGraph.__name__} class")

            self._instance = self._from_ts(raw_trans_sys,
                                           graph_name,
                                           config_yaml,
                                           save_flag, plot,
                                           human_intervention,
                                           plot_raw_ts,
                                           finite,
                                           debug)
        elif from_file:
            self._instance._graph_yaml = self._from_yaml(config_yaml)

        return self._instance

    def _from_ts(self, raw_ts: FiniteTransSys,
                 graph_name: str,
                 config_yaml: str,
                 save_flag: bool = False,
                 plot: bool = False,
                 human_intervention: int = 1,
                 plot_raw_ts: bool = False,
                 finite: bool = False,
                 debug: bool = False):
        """
        Returns  a Two Player transition system give a transition system with nodes that belong to eve only.
        :param raw_ts:
        :param graph_name:
        :param config_yaml:
        :param save_flag:
        :param plot:
        :param human_intervention:
        :param plot_raw_ts:
        :param debug:
        :return: A concrete instance of the  FiniteTransSys with both human and system nodes
        """
        return self._instance.from_raw_ts(raw_ts=raw_ts,
                                          graph_name=graph_name,
                                          config_yaml=config_yaml,
                                          save_flag=save_flag,
                                          plot=plot,
                                          human_intervention=human_intervention,
                                          plot_raw_ts=plot_raw_ts,
                                          finite=finite,
                                          debug=debug)

    def _load_pre_built(self):
        """
        A method to load the _pre_built dict with function calls to built in functions that create an
         concrete instance of FiniteTransitionSystem

        effect: Updates the built-in _pre_built dict with their respective keys and function calls as values
        """
        self._pre_built.update({"three_state_ts": self._instance.get_three_state_ts})
        self._pre_built.update({"five_state_ts": self._instance.get_five_state_ts})

    def _from_built_in_ts(self,
                          ts_name: str,
                          graph_name: str,
                          config_yaml: str,
                          save_flag: bool,
                          debug: bool,
                          plot: bool,
                          human_intervention: int,
                          plot_raw_TS: bool):
        """
        Return a pre-built Transition system based on the name of the Transition system which should be a valid key in
         the pre_built dict
        :param ts_name: The name of the system
        :return:
        """
        try:
            func = self._pre_built[ts_name]

            return func(graph_name,
                        config_yaml,
                        save_flag=save_flag,
                        debug=debug,
                        plot=plot,
                        human_intervention=human_intervention,
                        plot_raw_ts=plot_raw_TS)
        except KeyError:
            raise KeyError(f"Make sure you enter the correct name to access the pre built TS."
                            f" The built TS names are : {[i for i in self._pre_built.keys()]}")

    def _from_yaml(self, config_file_name: str) -> dict:

        config_data = self.load_YAML_config_data(config_file_name)

        return config_data

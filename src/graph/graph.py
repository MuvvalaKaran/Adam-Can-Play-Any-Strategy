import abc
import networkx as nx
import yaml
import os
import warnings


from graphviz import Digraph
from typing import List, Tuple, AnyStr
from helper_methods import deprecated


class Graph(abc.ABC):
    def __init__(self, config_yaml, graph, save_flag: bool=False):
        # self._filename: str = filename
        self._graph_yaml = None
        self._config_yaml: str = config_yaml
        self._save_flag: bool = save_flag
        self._graph: nx.MultiDiGraph = graph

    @abc.abstractmethod
    def construct_graph(self):
        pass

    @abc.abstractmethod
    def fancy_graph(self):
        pass

    @staticmethod
    def _get_current_working_directory() -> str:
        """
        A method to return the path of the current script
        NOTE : Verify what this function exactly returns
        :return: A path to the script we are running
        """
        # return os.path.dirname(os.path.realpath(__file__))
        return "/home/karan-m/Documents/Research/variant_1/Adam-Can-Play-Any-Strategy/src/"
    
    def read_yaml_file(self) -> None:
        """
        Reads the configuration yaml file @self._config_yaml associated with graph of
        type Networkx.LabelledDiGraph and store it in @self._graph_yaml
        :return:
        """
        if self._config_yaml is not None:
            file_name: str = self._config_yaml + ".yaml"
            file_add = Graph._get_current_working_directory() + file_name
            try:
                with open(file_add, 'r') as stream:
                    graph_data = yaml.load(stream, Loader=yaml.Loader)
            
            except FileNotFoundError as error:
                print(error)
                print(f"The file {file_name} does not exist")
            
            self._graph_yaml = graph_data['graph']

    def save_dot_graph(self, dot_object: Digraph, graph_name: str, view: bool = False) -> None:
        """
        A method to save the plotted graph in the respective folder
        :param dot_object: object of @Diagraph
        :param graph_name: String to identity the name by which the graph is saved as
        :param view: flag for viewing the object
        """
        if view:
            dot_object.view(cleanup=True)

        dot_object.render(Graph._get_current_working_directory() + f'/graph/{graph_name}', view=view, cleanup=True)

    def dump_to_yaml(self) -> None:
        """
        A method to dump the contents of the @self._graph in to @self._file_name yaml document which the Graph()
        class @read_yaml_file() reads to visualize it. By convention we dump files into config/file_name.yaml file.

        A sample dump should looks like this :

        >>> graph :
        >>>    vertices:
        >>>             tuple
        >>>             {'player' : 'eve'/'adam'}
        >>>    edges:
        >>>        parent_node, child_node, edge_weight
        """

        data = dict(
            graph=dict(
                vertices=[node for node in self._graph.nodes.data()],
                edges=[edge for edge in self._graph.edges.data()]
            )
        )

        config_file_name: str = str(self._config_yaml + '.yaml')
        config_file_add = Graph._get_current_working_directory() + config_file_name
        try:
            with open(config_file_add, 'w') as outfile:
                yaml.dump(data, outfile, default_flow_style=False)
        except FileNotFoundError:
            print(FileNotFoundError)
            print(f"The file {config_file_name} could not be found")

    def plot_graph(self):
        """
        A helper method to dump the graph data to a yaml file, read the yaml file and plotting the graph itself
        :return: None
        """
        # dump to yaml file
        self.dump_to_yaml()
        # read the yaml file
        self.read_yaml_file()
        # plot it
        self.fancy_graph()

    def add_state(self, state_name: nx.nodes, **kwargs) -> None:
        """
        A function to add states to a given graph
        :param state_name: The name associated with each node. As long as the type is Hashable, Networkx will not throw
        an error. This includes strings, numbers, tuples of strings and numbers etc. Definitetly not lists.
        :param kwargs: - Set or change node attributes using key=value

        Sample :
        >>> G = Graph
        >>> G.add_state(1)
        >>> G.add_state('v1', weight=0.4)
        """
        self._graph.add_node(state_name, **kwargs)

    def add_states_from(self, states: List, **kwargs) -> None:
        """
        A function to add state from a list to a given graph
        :param states: A container of nodes(list, dict, set etc.) OR a container of (node, attribute dict) tuples.
        :param kwargs: Update attributes for all nodes in nodes.

        Sample :
        >>> G = Graph
        >>> G.add_states_from(['v1', 1, 'Hello', ('v1', 2)])
        >>> G.add_states_from(['3' ,'v4'], player='eve')
        """
        self._graph.add_nodes_from(states, **kwargs)

    def add_state_attribute(self, state, attribute_key: str, attribute_value) -> None:
        """
        A function to add an attribute associated with a state
        :param state: A valid state of the graph @self._graph
        :param attribute_key: The name of attribute to be added
        :param attribute_value: The value associated with the attribute

        Sample:
        >>> G = Graph
        >>> G.add_state('v1')
        >>> G.add_state_attribute('v1', 'player', 'eve')
        >>> G.nodes['v1']
        TODO : Verify the output
        (v1, {'player', 'eve'})
        """
        if not isinstance(attribute_key, str):
            warnings.warn(f"The attribute key {attribute_key} is not of type string. I don't know how Networkx handles "
                          f"a non-string type dictionary key")

        self._graph.nodes[state][attribute_key] = attribute_value

    def add_state_attributes_from(self, states: List, attribute_key: str, attribute_value) -> None:
        """
        A helper function to add all the states with the same attribute_key and value pair
        :param states: A container of valid states of the graph @self._graph
        :param attribute_key: The name of attribute to be added
        :param attribute_value: The value associated with the attribute

        Sample:
        >>> G = Graph
        >>> G.add_state('v1')
        >>> G.add_state_attribute(['v1', 'v3', 'v4'], 'player', 'eve')
        """

        for _s in states:
            self.add_state_attribute(_s, attribute_key=attribute_key, attribute_value=attribute_value)

    def get_states(self) -> List:
        """
        A function to get all the states associated with a graph @self._graph
        Sample :
        >>> G = Graph
        >>> G.add_state(1)
        >>> G.add_states_from([2, 4])
        >>> list(G.nodes)
        [1, 2, 4]
        :return: A list of nodes corresponding to @self._graph
        """
        return list(self._graph.nodes)

    def get_states_w_attributes(self) -> List:
        """
        A function to get all the states with their respective attributes, if any
        Sample :
        >>> G = Graph
        >>> G.add_node('v1', player='eve')
        >>> G.add_node('v3')
        >>> G.nodes['v2']['accepting'] = True
        >>> list(G.nodes(data=True))
        [(v1, {'player':'eve'}), (v2, {'accepting': True}), (v3, {})]
        :return:
        """
        return list(self._graph.nodes(data=True))

    def get_state_w_attribute(self, state, attribute: str):
        """
        A function to get an attribute associates with a state
        TODO: Verify this
        :param state: A valid node of the graph. If the node does not exist then the graph throws an error I guess
        :param attribute: A valid attribute associated with a node of the graph @self._graph. If no such attribute
        exists then we return None
        :return: The value associated with a node attribute

        Sample :
        >>> G = Graph
        >>> G.add_state('1', weight=3)
        >>> G.nodes['1'].get('weight')
        3
        >>> G.nodes['1'].get('player')
        None
        >>> G.nodes['2']
        ERROR
        """

        try:
            r_val = self._graph.nodes[state].get(attribute)
            if r_val is None:
                warnings.warn(f"WARNING: The state {state} does not have any attribute {attribute}")

            return r_val

        except KeyError as error:
            print(error)
            print(f"The state {state} does not exist in the graph {self._graph.__getattribute__('name')}")

    def add_edge(self, u, v, **kwargs) -> None:
        """
        A function to add AN edge between u and v.

        NOTE: The nodes u and v will be automatically added if they are not already in the graph.
        Edge attributes can be specified with keywords or by directly accessing the edge's attribute dictionary
        :param u: The node from where the edge originated from
        :param v: The node from where the edge goes to
        :param kwargs: Edge data (or labels or objects) can be assigned using keyword argument.

        Sample :
        >>> G = Graph
        >>> G.add_states_from(['v1', 'v2'])
        NOTE : I deliberately left out state 'v3' to show to that the state is automatically added if not specified
        before
        # lets build this graph

        v1 <----> v2 --(2)-->v3 # (weight of 2 corresponding to the edge v2 to v3)

        >>> G.add_edge('v1', 'v2')
        >>> G.add_edge('v2', 'v1')
        >>> G.add_edge('v2', 'v3', weight=2)

        NOTE : You can also use other attributes like ('v2', 'v3', second_weight=10, edge_thickness=2)
        """
        self._graph.add_edge(u, v, **kwargs)

    def add_edges_from(self, edges: List[Tuple], **kwargs) -> None:
        """
        A function to add all the edges in @edges (container of edges)
        :param edges: Each edge in @edges will be added to the graph. The edges could be a tuple of 2-(u, v) or 3-
        (u, v, d). Here d is dictionary containing edge data
        :param kwargs: Edge data (or label or object) can be assigned using keyword arguments.

        Sample:
        >>> G = Graph
        >>> G.add_edges_from([(1, 2), ('v1', 'v2'), (('v1', 2), ('v2', 3))])
        >>> G.add_edges_from([(3, 4), ("Hello", 'v3')], label='!b & c')
        """
        self._graph.add_nodes_from(edges, **kwargs)

    def add_weighted_edges_from(self, edges_w_weight: List[Tuple]):
        """
        A function to add weighted edges in @edges_w_weights with specified weight attribute
        :param edges_w_weight: A container of edges of the form - a tuple of 3 (u, v, w) where w the attribute weight
        NOTE : the value associated with the weight attribute does not need to be a number.

        Sample:
        >>> G = Graph
        >>> G.add_weighted_edges_from([(1, 3, 7), ('v1', 'v2', 1)])
        """
        self._graph.add_weighted_edges_from(edges_w_weight)

    @deprecated
    def add_edge_attributes(self, u, v, attribute_key, attribute_value):
        self.add_edge(u, v, attribute_key=attribute_value)

    def get_edge_attributes(self, u, v, attribute: str):
        """
        A function to get an attribute associated with an edge (u, v)
        :param u: The initial state from which the edge originates from
        :param v: The ending states at which the edge terminates at
        :param attribute: An attribute associated with the given edge
        :return: The value associated with the attribute of an edge (u, v)

        Sample:
        >>> G = Graph
        >>> G.add_states_from(['v1', 'v2', 'v3'])
        >>> G.add_edge('v1', 'v2', weight=3)
        >>> G.get_edge_attributes('v1', 'v2', 'weight')
        3
        """

        edge_attr = self._graph[u][v][0].get(attribute)

        if edge_attr is None:
            warnings.warn(f"The edge from {u}-->{v} does not contain the attribute {attribute}")

        return edge_attr

    def get_transitions(self) -> List:
        """
        A function to get all the transitions associated with a graph
        :return: A list of edges
        """
        return self._graph.edges.data()

    def get_edge_weight(self, u, v):
        """
        A function to get the weight associated with an edge. This method calls @get_edge_attributes(attribute=weight).
        So, if the edge (u,v) does not have a weight associated with it, then it will throw an error.
        :param u:
        :param v:
        :return:
        """
        return self.get_edge_attributes(u, v, 'weight')

    def get_adj_nodes(self):
        pass

    def add_initial_state(self, state) -> None:
        """
        A function to add the 'init' attribute to a given state
        TODO: What if there does not exist that state. What does networkx throw in that case? Add try-exception code
        here
        :param state: A valid state that belongs to @self._graph
        """
        self._graph.nodes[state]['init'] = True

    def add_initial_states_from(self, states: List) -> None:
        """
        A function to add the 'init' attribute to a bunch of states in @states (container of states)
        :param states: A container like list, tuple, set etc. containing a bunch of initial states
        """
        for _s in states:
            self._graph.nodes[_s]['init'] = True

    def get_initial_states(self):
        """
        A function to get the initial state or a set of initial states (if multiple)
        :return: a list of state - # of elements >= 0
        """
        _init_state = []

        for n in self._graph.nodes.data('init'):
            if n[1] is True:
                _init_state.append(n)

        if len(_init_state) == 0:
            warnings.warn("WARNING: The set of initial states is empty. Returning an empty list.")

        return _init_state

    def add_accepting_state(self, state) -> None:
        """
        A function to add the 'accepting' attribute to a given state
        TODO: What if there does not exist that state. What does networkx throw in that case? Add try-exception code
        here
        :param state: A valid state that belongs to @self._graph
        """
        self._graph.nodes[state]['accepting'] = True

    def add_accepting_states_from(self, states: List) -> None:
        """
        A function to add the 'accepting' attribute to a bunch of states in @states (container of states)
        :param states: A container like list, tuple, set etc. containing a bunch of accepting states
        """
        for _s in states:
            self._graph.nodes[_s]['accepting'] = True

    def get_accepting_states(self):
        """
        A function to get the accepting state or a set of accepting states (if multiple)
        :return: a list of state - # of elements >= 0
        """
        _accp_state = []

        for n in self._graph.nodes.data('accepting'):
            if n[1] is True:
                _accp_state.append(n)

        if len(_accp_state) == 0:
            warnings.warn("WARNING: The set of accepting states is empty. Returning an empty list.")

        return _accp_state

    def __str__(self):
        g = ('Graph : ' + self._graph.__getattribute__("name") + '\n' +
             'Players : ' + self.player + '\n' +
             'states : ' + self.get_states() + '\n' +
             'initial state : ' + self.get_initial_states() + '\n' +
             'accepting state : ' + self.get_accepting_states() + '\n' +
             'Transitions : ' + self.get_transitions() + '\n')

        return g


class TwoPlayerGraph(Graph):
    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False):
        # initialize the Graph class instance variables
        self._config_yaml = config_yaml
        self._save_flag = save_flag
        self._graph_name = graph_name
        
    def construct_graph(self):
        two_player_graph: nx.MultiDiGraph = nx.MultiDiGraph(name=self._graph_name)
        # add this graph object of type of Networkx to our Graph class 
        self._graph = two_player_graph

    def fancy_graph(self, color=("lightgrey", "red", "purple")) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["vertices"]
        for n in nodes:
            # default color for all the nodes is grey
            dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[0]})
            if n[1].get('init'):
                # default color for init node is red
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[1]})
            if n[1].get('accepting'):
                # default color for accepting node is purple
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[2]})
            if n[1]['player'] == 'eve':
                dot.node(str(n[0]), _attributes={"shape": "rectangle"})
            else:
                dot.node(str(n[0]), _attributes={"shape": "circle"})
        
        # add all the edges
        edges = self._graph_yaml["edges"]

        # load the weights to illustrate on the graph
        for counter, edge in enumerate(edges):
            if edge[2].get('strategy') is True:
                dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2]['weight']), _attributes={'color': 'red'})
            else:
                dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2]['weight']))

        # set graph attributes
        # dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            graph_name = str(self._graph.__getattribute__('name'))
            self.save_dot_graph(dot, graph_name, True)


class GminGraph(TwoPlayerGraph):
    pass

class GmaxGraph(TwoPlayerGraph):
    pass


class FiniteTransSys(TwoPlayerGraph):
    pass


class DFAGraph(Graph):

    def __init__(self, filename, config_yaml, graph, save_flag: bool = False):
        # super().__init__(config_yaml, graph, save_flag)
        pass

    def construct_graph(self):
        pass

    def plot_graph(self):
        pass


if __name__ == "__main__":
    two_player_graph = TwoPlayerGraph('sample_graph', 'config/graph', save_flag=True)
    two_player_graph.construct_graph()

    two_player_graph.add_states_from(['v1', 'v2', 'v3'])
    two_player_graph.add_weighted_edges_from([('v1', 'v2', '1'),
                                              ('v2', 'v1', '2'),
                                              ('v1', 'v3', '1'),
                                              ('v3', 'v3', '0.5')])

    two_player_graph.add_state_attribute('v1', 'player', 'eve')
    two_player_graph.add_state_attribute('v2', 'player', 'adam')
    two_player_graph.add_state_attribute('v3', 'player', 'adam')

    two_player_graph.add_state_attributes_from(['v1', 'v2', 'v3'], 'accepting', True)
    two_player_graph.plot_graph()


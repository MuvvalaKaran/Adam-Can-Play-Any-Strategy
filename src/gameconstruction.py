import networkx as nx
import yaml
import copy
import random
import os

from helper_methods import deprecated
from typing import List, Tuple, AnyStr
from graphviz import Digraph
from src.PayoffFunc import PayoffFunc

print_edge = False
# use this boolean to print nodes and edges with their respective weights
print_Gmin_nodes = False
print_Gmax_nodes = False
print_Gmin_edges = False
print_Gmax_edges = False
# print a particular strategy with m memory
print_str_m = False
# print strategies for a given range of m
print_range_str_m = False
# set this boolean to true if you want to test Gmin and Gmax construction on a smaller graph
test_case = True

# test inf payoff function
test_inf = True
# test sup payoff function
test_sup = False


def get_cwd_path() -> AnyStr:
    return os.path.dirname(os.path.realpath(__file__))


class Strategy(object):
    def __init__(self, path: List, player: str) -> None:
        """
        A class to hold all the strategies
        :param path:
        :param player:
        """
        self.path = path
        self.player = player
        self.init = None
        # dictionary of form "m" : "all paths"
        # self.dict = {}

    def setinit(self, init: bool) -> None:
        '''
        Init flag set to be True for strategies begining from initial vertex
        :param init: Boolean variable idicating strategies starting from the init vertex
        :return: None
        '''
        self.init = init

    def setpath(self, newpath: List) -> None:
        self.path = newpath

    def setplayer(self, player: List) -> None:
        self.player = player

    # def updatedict(self, key, value):
    #     self.dict.update({key: value})

class Graph(object):

    def __init__(self, save_flag: bool = True) -> None:
        self.file_name: str = None
        self.graph_yaml = self.read_yaml_file(self.file_name)
        self.save_flag = save_flag
        self.graph = None
        self.Strs_dict = {}

    @staticmethod
    def read_yaml_file(config_files: str) -> nx:
        """
            reads the yaml file data and stores them in appropriate variables
        """
        graph = None

        if config_files is not None:
            try:
                with open(config_files + ".yaml", 'r') as stream:
                    data_loaded = yaml.load(stream, Loader=yaml.Loader)
            except FileNotFoundError:
                file_name = config_files + ".yaml"
                print(FileNotFoundError)
                print(f"The file {file_name} does not exist")

            graph = data_loaded['graph']

        return graph

    def plot_fancy_graph(self, color=("lightgrey", "red", "purple")) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self.graph_yaml["vertices"]
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
        edges = self.graph_yaml["edges"]

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

        if self.save_flag:
            graph_name = str(self.graph.__getattribute__('name'))
            self.save_dot_graph(dot, graph_name, True)

    def save_dot_graph(self, dot_object: Digraph, graph_name: str, view: bool = False) -> None:
        """
        A method to save the plotted graph in the respective folder
        :param dot_object: object of @Diagraph
        :param view: flag for viewing the object
        """
        if view:
            dot_object.view(cleanup=True)

        dot_object.render(get_cwd_path() + f'/graph/{graph_name}', view=view, cleanup=True)

    def create_multigrpah(self, reward=False) -> nx.MultiDiGraph:
        """
        Method to create a multigraph
        :return: Multigraph constructed in networkx
        :rtype: @Digraph
        """
        MG: nx.MultiDiGraph = nx.MultiDiGraph(name="org_graph")

        if test_case:
            # MG.add_nodes_from(['v1', 'v2', 'v3'])
            # MG.add_weighted_edges_from([('v1', 'v2', '1'),
            #                             ('v2', 'v1', '2'),
            #                             ('v1', 'v3', '1'),
            #                             ('v3', 'v3', '0.5')
            #                             ])
            #
            # # assign each node a player - this is then later used to plot them conveniently
            # MG.nodes[1]['player'] = 'eve'
            # MG.nodes[2]['player'] = 'adam'
            # MG.nodes[3]['player'] = 'adam'
            MG.add_nodes_from(['v1', 'v2', 'v3', 'v4', 'v5',
                               'v6', 'v7', 'v8', 'v9', 'v10',
                               'v11', 'v12', 'v13', 'v14', 'v15',
                               'v16', 'v17', 'v18', 'v19', 'v20',
                               'v21', 'v22', 'v23', 'v24', 'v25',
                               'v26', 'v27', 'v28', 'v29', 'v30',
                               'v31', 'v32', 'v33', 'v34', 'v35',
                               'v36', 'v37', 'v38', 'v39', 'v40'])
            # cost based edges
            if not reward:
                s12: str = str(random.randint(1, 9))
                s21: str = str(random.randint(1, 9))
                s23: str = str(random.randint(1, 9))
                s33: str = str(1)
                print(f"Values of s12 : {s12}, s21: {s21}, s23: {s23}, s33: {s33}")
                MG.add_weighted_edges_from([('v1', 'v32', s12),  # region q_2
                                            ('v2', 'v3', s12), ('v2', 'v8', '0'), ('v2', 'v10', '0'),
                                            ('v3', 'v14', s21), ('v3', 'v16', s23),
                                            ('v4', 'v1', s21), ('v4', 'v9', '0'), ('v4', 'v10', '0'),
                                            ('v5', 'v27', s33),
                                            ('v6', 'v5', s23), ('v6', 'v8', '0'), ('v6', 'v9', '0'),
                                            ('v7', 'v5', s33), ('v7', 'v8', '0'), ('v7', 'v9', '0'),
                                            ('v8', 'v39', s12),
                                            ('v9', 'v8', s21), ('v9', 'v20', s23),
                                            ('v10', 'v30', s33),
                                            ('v11', 'v12', s12),  # region q_1 starts
                                            ('v12', 'v13', s12), ('v12', 'v18', '0'), ('v12', 'v20', '0'),
                                            ('v13', 'v16', s23), ('v13', 'v14', s21),
                                            ('v14', 'v11', s12), ('v14', 'v19', '0'), ('v14', 'v20', '0'),
                                            ('v15', 'v27', s33),
                                            ('v16', 'v15', s23), ('v16', 'v18', '0'), ('v16', 'v19', '0'),
                                            ('v17', 'v15', s33), ('v17', 'v18', '0'), ('v17', 'v19', '0'),
                                            ('v18', 'v19', s12),
                                            ('v19', 'v18', s21), ('v19', 'v20', s23),
                                            ('v20', 'v30', s33),
                                            ('v21', 'v22', s12),  # region q_0 starts
                                            ('v22', 'v23', s12), ('v22', 'v28', '0'), ('v22', 'v30', '0'),
                                            ('v23', 'v26', s23), ('v23', 'v24', s21),
                                            ('v24', 'v21', s21), ('v24', 'v29', '0'), ('v24', 'v30', '0'),
                                            ('v25', 'v27', s33),
                                            ('v26', 'v25', s23), ('v26', 'v28', '0'), ('v26', 'v29', '0'),
                                            ('v27', 'v25', s33), ('v27', 'v28', '0'), ('v27', 'v29', '0'),
                                            ('v28', 'v29', s12),
                                            ('v29', 'v28', s21), ('v29', 'v30', s23),
                                            ('v30', 'v30', s33),
                                            ('v31', 'v32', s12),  # region q_4 starts
                                            ('v32', 'v33', s12), ('v32', 'v38', '0'), ('v32', 'v40', '0'),
                                            ('v33', 'v36', s23), ('v33', 'v34', s21),
                                            ('v34', 'v31', s21), ('v34', 'v39', '0'), ('v34', 'v40', '0'),
                                            ('v35', 'v37', s33),
                                            ('v36', 'v35', s23), ('v36', 'v38', '0'), ('v36', 'v39', '0'),
                                            ('v37', 'v35', s33), ('v37', 'v38', '0'), ('v37', 'v39', '0'),
                                            ('v38', 'v39', s12),
                                            ('v39', 'v38', s21), ('v39', 'v40', s23),
                                            ('v40', 'v40', s33)])
            # reward based edges
            else:
                MG.add_weighted_edges_from([('v1', 'v3', '1'),
                                            ('v1', 'v2', '0'),
                                            ('v2', 'v1', '0'), ('v2', 'v3', '0'), ('v2', 'v5', '0'), ('v2', 'v7', '0'),
                                            ('v2', 'v9', '0'),
                                            ('v3', 'v1', '1'),
                                            ('v3', 'v5', '2'),
                                            ('v3', 'v4', '0'),
                                            ('v4', 'v3', '0'), ('v4', 'v1', '0'), ('v4', 'v5', '0'), ('v4', 'v7', '0'),
                                            ('v4', 'v9', '0'),
                                            ('v5', 'v3', '2'),
                                            ('v5', 'v7', '3'),
                                            ('v5', 'v6', '0'),
                                            ('v6', 'v5', '0'), ('v6', 'v3', '0'), ('v6', 'v1', '0'), ('v6', 'v7', '0'),
                                            ('v6', 'v9', '0'),
                                            ('v7', 'v10', '4'),
                                            ('v7', 'v11', '4'),
                                            ('v7', 'v13', '4'),
                                            ('v7', 'v15', '0'),
                                            # ('v10', 'v16', '6'),
                                            # ('v11', 'v16', '6'),
                                            # ('v12', 'v16', '6'),
                                            # ('v13', 'v16', '6'),
                                            # ('v14', 'v10', '6'), ('v14', 'v11', '6'), ('v14', 'v13', '6'), ('v14', 'v15', '6'),
                                            # ('v15', 'v14', '6'), ('v15', 'v12', '6'), ('v15', 'v13', '6'), ('v15', 'v10', '6'), ('v15', 'v11', '6'),
                                            ('v10', 'v16', '5'),
                                            ('v11', 'v16', '5'),
                                            ('v12', 'v16', '5'),
                                            ('v13', 'v16', '5'),
                                            ('v14', 'v10', '5'), ('v14', 'v11', '5'), ('v14', 'v13', '5'),
                                            ('v14', 'v15', '0'),
                                            ('v15', 'v14', '0'), ('v15', 'v12', '0'), ('v15', 'v13', '0'),
                                            ('v15', 'v10', '0'), ('v15', 'v11', '0'),
                                            ('v9', 'v16', '3'),
                                            ('v16', 'v16', '0')])  # self-loop edge

            # assign each node a player - this is then later used to plot them conveniently
            MG.nodes['v1']['player'] = 'eve'
            MG.nodes['v2']['player'] = 'adam'
            MG.nodes['v3']['player'] = 'eve'
            MG.nodes['v4']['player'] = 'adam'
            MG.nodes['v5']['player'] = 'eve'
            MG.nodes['v6']['player'] = 'adam'
            MG.nodes['v7']['player'] = 'adam'
            MG.nodes['v8']['player'] = 'eve'
            MG.nodes['v9']['player'] = 'eve'
            MG.nodes['v10']['player'] = 'eve'
            MG.nodes['v11']['player'] = 'eve'
            MG.nodes['v12']['player'] = 'adam'
            MG.nodes['v13']['player'] = 'eve'
            MG.nodes['v14']['player'] = 'adam'
            MG.nodes['v15']['player'] = 'eve'
            MG.nodes['v16']['player'] = 'adam'
            MG.nodes['v17']['player'] = 'adam'
            MG.nodes['v18']['player'] = 'eve'
            MG.nodes['v19']['player'] = 'eve'
            MG.nodes['v20']['player'] = 'eve'
            MG.nodes['v21']['player'] = 'eve'
            MG.nodes['v22']['player'] = 'adam'
            MG.nodes['v23']['player'] = 'eve'
            MG.nodes['v24']['player'] = 'adam'
            MG.nodes['v25']['player'] = 'eve'
            MG.nodes['v26']['player'] = 'adam'
            MG.nodes['v27']['player'] = 'adam'
            MG.nodes['v28']['player'] = 'eve'
            MG.nodes['v29']['player'] = 'eve'
            MG.nodes['v30']['player'] = 'eve'
            MG.nodes['v31']['player'] = 'eve'
            MG.nodes['v32']['player'] = 'adam'
            MG.nodes['v33']['player'] = 'eve'
            MG.nodes['v34']['player'] = 'adam'
            MG.nodes['v35']['player'] = 'eve'
            MG.nodes['v36']['player'] = 'adam'
            MG.nodes['v37']['player'] = 'adam'
            MG.nodes['v38']['player'] = 'eve'
            MG.nodes['v39']['player'] = 'eve'
            MG.nodes['v40']['player'] = 'eve'

            # add accepting states to the graph
            MG.nodes['v21']['accepting'] = True
            MG.nodes['v22']['accepting'] = True
            MG.nodes['v23']['accepting'] = True
            MG.nodes['v24']['accepting'] = True
            MG.nodes['v25']['accepting'] = True
            MG.nodes['v26']['accepting'] = True
            MG.nodes['v27']['accepting'] = True
            MG.nodes['v28']['accepting'] = True
            MG.nodes['v29']['accepting'] = True
            MG.nodes['v30']['accepting'] = True

            # add node 1 as the initial node
            MG.nodes['v3']['init'] = True

            self.graph = MG

        else:
            MG.add_nodes_from(['v1', 'v2', 'v3', 'v4', 'v5'])
            if not reward:
                # personal test case graph with same no. of nodes (5)
                MG.add_weighted_edges_from([('v2', 'v3', '2'),
                                            ('v3', 'v2', '2'),
                                            ('v2', 'v1', '3'),
                                            ('v3', 'v1', '3'),
                                            ('v1', 'v1', '3'),
                                            ('v3', 'v4', '2'),
                                            ('v2', 'v4', '2'),
                                            ('v4', 'v5', '1'),
                                            # ('v4', 'v1', '0'),
                                            # ('v5', 'v4', '6'),
                                            ('v5', 'v5', '0')
                                            ])
            # reward based edges
            else:
                MG.add_weighted_edges_from([('v2', 'v3', '2'),
                                            ('v3', 'v2', '2'),
                                            ('v2', 'v1', '1'),
                                            ('v3', 'v1', '1'),
                                            ('v1', 'v1', '0'),
                                            ('v3', 'v4', '2'),
                                            ('v2', 'v4', '2'),
                                            ('v4', 'v5', '3'),
                                            # ('v4', 'v1', '0'),
                                            # ('v5', 'v4', '6'),
                                            ('v5', 'v5', '4')
                                            ])
            # original graph
            # MG.add_weighted_edges_from([('v1', 'v2', '1'),
            #                             ('v2', 'v1', '-1'),
            #                             ('v1', 'v3', '1'),
            #                             # ('v2', 'v3', '1'),
            #                             ('v3', 'v3', '0.5'),
            #                             ('v3', 'v5', '1'),
            #                             ('v2', 'v4', '2'),
            #                             ('v4', 'v4', '2'),
            #                             # ('v4', 'v1', '0'),
            #                             # ('v5', 'v4', '6'),
            #                             ('v5', 'v5', '1')
            #                             ])

            # assign each node a player - this is then later used to plot them conveniently
            MG.nodes['v1']['player'] = 'eve'
            MG.nodes['v2']['player'] = 'eve'
            MG.nodes['v3']['player'] = 'adam'
            MG.nodes['v4']['player'] = 'eve'
            MG.nodes['v5']['player'] = 'eve'


        # NOTE: the data player and the init flag cannot be accessed as graph[node]['init'/ 'player'] you have to
        #  first access the data as graph.nodes.data() and loop over the list; each element in that list is a tuple
        #  (NOT A DICT) of the form (node_name, {key: value})

            # add node 1 as the initial node
            MG.nodes['v2']['init'] = True

            # add accepting states to the graph
            MG.nodes['v5']['accepting'] = True
            self.graph = MG

        return MG

    def print_edges(self) -> None:
        for (u, v, wt) in self.graph.edges.data('weight'):
            print(f"({u}, {v}, {wt})")

    def dump_to_yaml(self) -> None:
        """
        A method to dump the contents of the grpah in to yaml document which the Graph() class reads to visualize it

        The sample dump should look like

        graph :
            vertices:
                     int
                     {'player' : eve/adam}
            edges:
                (parent node, child node, edge weight)

        :param graph:
        :param file_name: Name of the original graph yaml
        :return: None
        """

        data = dict(
            graph=dict(
                vertices=[node for node in self.graph.nodes.data()],
                edges=[edge for edge in self.graph.edges.data()]
            )
        )

        config_file_name: str = str(self.file_name + '.yaml')
        try:
            with open(config_file_name, 'w') as outfile:
                yaml.dump(data, outfile, default_flow_style=False)
        except FileNotFoundError:
            print(FileNotFoundError)
            print(f"The file {config_file_name} could not be found")

    def construct_Gmin(self, org_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Method to construct a Gmin graph such that for payoff function Sup : Reg(G) = Reg(Gmin)

        V' = V x {w(e) | e belongs to the original set of Edges of G}
        V'e = {(v, n) belong to V' | v belongs to Ve(vertices that belong to Eve)}
        v'i = (vi, W) where W is the maximum weight value in G
        An edge exists ((u, n), (v, m)) iff (u, v) belong to E and m = min{n, w(u, v)}
        w'((u, n), (v, m)) = m
        :return: Graph Gmin
        """

        # create a new MG graph
        Gmin: nx.MultiDiGraph = nx.MultiDiGraph(name="Gmin_graph")

        # construct new set of states V'
        V_prime = [(v, str(w)) for v in org_graph.nodes.data() for _, _, w in org_graph.edges.data('weight')]

        # find the maximum weight in the og graph(G)
        # specifically adding self.graph.edges.data('weight') to a create to tuple where the
        # third element is the weight value
        max_edge = max(dict(org_graph.edges).items(), key=lambda x: x[1]['weight'])
        W: str = max_edge[1].get('weight')

        # assign nodes to Gmin with player as attributes to each node
        for n in V_prime:

            if n[0][1]['player'] == "eve":
                Gmin.add_node((n[0][0], n[1]))
                Gmin.nodes[(n[0][0], n[1])]['player'] = 'eve'
            else:
                Gmin.add_node((n[0][0], n[1]))
                Gmin.nodes[(n[0][0], n[1])]['player'] = 'adam'

            # if the node has init attribute and n[1] == W then add it to the init vertex in Gmin
            if n[0][1].get('init') and n[1] == W:
                Gmin.nodes[(n[0][0], n[1])]['init'] = True

        if print_Gmin_nodes:
            print("Printing Gmin nodes : \n", Gmin.nodes.data())

        # constructing edges as per the requirement mentioned in the doc_string
        for parent in Gmin.nodes:
            for child in Gmin.nodes:
                if org_graph.has_edge(parent[0], child[0]):
                    if float(child[1]) == min(float(parent[1]), float(org_graph.get_edge_data(parent[0], child[0])[0]['weight'])):
                        Gmin.add_edge(parent, child, weight=child[1])

        if print_Gmin_edges:
            print("Printing Gmin edges and with weights \n")
            for (u, v, wt) in Gmin.edges.data('weight'):
                print(f"({u}, {v}, {wt})")

        return Gmin

    def construct_Gmax(self, org_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Method to construct a Gmax graph such that for payoff function Sup : Reg(G) = Reg(Gmax)

        V' = V x {w(e) | e belongs to the original set of Edges of G}
        V'e = {(v, n) belong to V' | v belongs to Ve(vertices that belong to Eve)}
        v'i = (vi, W) where W is the maximum weight value in G
        An edge exists ((u, n), (v, m)) iff (u, v) belong to E and m = max{n, w(u, v)}
        w'((u, n), (v, m)) = m
        :return:
        """

        # create a new MG graph
        Gmax: nx.MultiDiGraph = nx.MultiDiGraph(name="Gmax_graph")

        # construct new set of states V'
        V_prime = [(v, str(w)) for v in org_graph.nodes.data() for _, _, w in org_graph.edges.data('weight')]

        # find the maximum weight in the og graph(G)
        # specifically adding self.graph.edges.data('weight') to a create to tuple where the
        # third element is the weight value
        max_edge = max(dict(org_graph.edges).items(), key=lambda x: x[1]['weight'])
        W: str = max_edge[1].get('weight')

        # assign nodes to Gmax with player as attributes to each node
        for n in V_prime:

            if n[0][1]['player'] == "eve":
                Gmax.add_node((n[0][0], n[1]))
                Gmax.nodes[(n[0][0], n[1])]['player'] = 'eve'
            else:
                Gmax.add_node((n[0][0], n[1]))
                Gmax.nodes[(n[0][0], n[1])]['player'] = 'adam'

            # if the node has init attribute and n[1] == W then add it to the init vertex in Gmin
            if n[0][1].get('init') and n[1] == W:
                Gmax.nodes[(n[0][0], n[1])]['init'] = True

        if print_Gmax_nodes:
            print("Printing Gmax nodes : \n", Gmax.nodes.data())

        # constructing edges as per the requirement mentioned in the doc_string
        for parent in Gmax.nodes:
            for child in Gmax.nodes:
                if org_graph.has_edge(parent[0], child[0]):
                    if float(child[1]) == max(float(parent[1]), float(org_graph.get_edge_data(parent[0], child[0])[0]['weight'])):
                        Gmax.add_edge(parent, child, weight=child[1])

        if print_Gmax_edges:
            print("Printing Gmax edges and with weights \n")
            for (u, v, wt) in Gmax.edges.data('weight'):
                print(f"({u}, {v}, {wt})")

        return Gmax

    # helper method to get states that belong to eve and adam respectively
    def get_eve_adam_states(self, graph: nx.MultiDiGraph) -> Tuple[list, list]:
        """
        A method to retrieve the states that belong to eve and adam
        :param graph:
        :type graph: netowkrx
        :return: (eve_states, adam_state)
        """

        eve_states: List = []
        adam_states: List = []

        for n in graph.nodes.data():
            if n[1]['player'] == 'eve':
                eve_states.append(n)
            else:
                adam_states.append(n)

        return eve_states, adam_states

    # use this method to create a range of strategies
    def create_set_of_strategies(self, graph: nx.MultiDiGraph, bound: int) -> dict:
        """
        Hypothetically G for eve and adam should be infinite. But technically we don't have infinite memory to compute
        strategies with infinite memory. Also we implement recursion to compute all possible paths. The max depth of
        recursion in python is (~1000). So m < 1000 is a must.

        This method is used to compute strategies for a given range of m.
        :param graph: Graph from which we would like to compute strategies for a range of m values
        :param bound: Upper Bound on the memory
        :return: a dictionary of form {{m:set of strategies from each vertex }}
        """

        assert (bound < 1000), "Please enter bound to be less than 1000 as it is max recursion depth"
        # trim all non-essential stuff
        states: Tuple[list, list] = self.get_eve_adam_states(graph)

        # create eve and adam list to hold the vertex labels
        _eve_states: List = []
        _adam_states: List = []

        # states[0] belong to eve and state[1] belong to adam
        for e in states[0]:
            _eve_states.append(e[0])

        for a in states[1]:
            _adam_states.append(a[0])

        for m in range(1, bound):
            strs = self.strategy_synthesis_w_finite_memory(graph, m, _eve_states, _adam_states)
            self.Strs_dict.update({m: strs})

        return self.Strs_dict

    def strategy_synthesis_w_finite_memory(self, graph: nx.MultiDiGraph, m: int, _eve_states: List, _adam_states: List)\
            -> dict:
        """
        A method to compute a set of strategies for a given graph. This method calls @compute_all_paths() to compute
        all possible paths from a give vertex. While doing so each path is assigned which player that strategy belongs
        to. In this method m = 1 denotes memoryless strategy and m = n denotes you roll out n times excluding the
        initial vertex. In total the path length is n + 1 as we include the initial vertex as well.

        We can employ this method to create a set of strategies for a given memory(m) value
        :param graph: Graph from which we would like to compute strategies
        :param m:memory of the strategy
        :param _eve_states:
        :param _adam_states:
        :return:a dictionary of all paths computes from each states with memory m {{vertex_label: paths} }
        """
        paths = {}

        for n in graph.nodes():
            paths.update({str(n): self.compute_all_path(graph, n, m,
                                                        _eve_states,
                                                        _adam_states,
                                                        pathx=Strategy([], None))})
        return paths

    def compute_all_path(self, graph: nx.MultiDiGraph, curr_node: Tuple, m: int, _eve_state: List, _adam_states: List, pathx) \
            -> List:
        """
        A method to compute all the paths possible from a given state. This function is called recursively until
        memory(m) becaome 0. So, technically we rollout m + 1 times. with the first vertex in the path being the vertex
        from where we begin compute paths
        :param graph: Graph from which we would like to compute all possible path from a vertex
        :param curr_node: Current node
        :param m: memory
        :param _eve_state: set of states that belong to eve
        :param _adam_states: set of states that belong to adam
        :param pathx: path that keeps getting appended with every recursion
        :return: a list of all paths from the given vertex. It includes self-loops as well
        """
        if type(m) is not int:
            raise ValueError

        # initially pathx is an empty list []
        path = copy.deepcopy(pathx)
        newpath = path.path + [curr_node]
        path.setpath(newpath)
        if not path.player:
            path.setplayer("eve" if path.path[0] in _eve_state else "adam")

        if not path.init:
            if path.path[0] == 1 or path.path[0] == (1, 2):
                path.setinit(True)
        paths = []
        if m != 0:
            for e in graph.edges(curr_node):
                node = e[1]
                # here we get away with using eve and adam states as none because the states are only used at
                # path initialization time
                newpaths = self.compute_all_path(graph, node, m-1, None, None, path)
                for newpath in newpaths:
                    paths.append(newpath)
        else:
            paths.append(path)
            return paths
        return paths

    def get_set_of_strategies(self) -> dict:
        """
        A getter method to get all the computed strategies for different memory values
        :return: The dict with key as  m (memory) and value all the possible plays
        """
        return self.Strs_dict

    def print_set_of_strategies(self) -> None:
        """
        A helper method to print all the strategies with the corresponding memory value
        """
        for k, v in self.Strs_dict.items():
            print(f"For memory {k} :")
            for vertex, pths in v.items():
                print(f"for vertex {vertex}, the number of paths is {len(pths)}")
            print("")


    def Gmin_to_G_play(self, play: List[Tuple]) -> List:
        """
        A helper method to get the corresponding play on the Gmin/Gmax grpah to G
        :param play:
        :return:
        """
        G_play = []

        for i in play:
            G_play.append(i[0])
        return G_play

    @deprecated
    def test_inf_and_liminf_limsup(self, gmin_graph: nx.MultiDiGraph, org_graph: nx.MultiDiGraph) -> None:
        print("Testing inf and LimInf and LimSup payoff for a play on Gmin and the respective play in G")

        # get a strategy from m = 100 for the org_graph
        eve_states, adam_states = self.get_eve_adam_states(gmin_graph)
        trail = self.strategy_synthesis_w_finite_memory(graph=gmin_graph, m=10,
                                                            _eve_states=eve_states,
                                                            _adam_states=adam_states)

        # compute imf and sup value for a play on graph say m = 10 and from vertex 1

        # get the sequence of edges for the give play
        w_min = []
        w = []
        # get key with init Flag = True
        # rn_vertex = random.choice(list(trail.keys()))
        init_vertex = str(('v1', '2'))
        rn_play = random.choice(list(trail.get(init_vertex)))
        # play = trail.get(rn_vertex)[rn_play].path
        print(rn_play.path)
        # get the corresponding play in org_graph
        g_play = self.Gmin_to_G_play(rn_play.path)
        print(g_play)
        # return
        for i in range(len(rn_play.path) - 1):
            w.append(org_graph.get_edge_data(g_play[i], g_play[i + 1])[0]['weight'])
            w_min.append(gmin_graph.get_edge_data(rn_play.path[i], rn_play.path[i + 1])[0]['weight'])
        # val_sup = PayoffFunc.Sup(w)
        val_inf = PayoffFunc.Inf(w)
        val_inf_min = PayoffFunc.Inf(w_min)
        print(f"val for payoff sup is {val_inf_min} for the given play: \n {g_play}")
        print(f"val for payoff inf is {val_inf} for the given play: \n {rn_play.path}")

        val_liminf = PayoffFunc.LimInf(w_min)
        val_limsup = PayoffFunc.LimSup(w_min)
        print(f"val for payoff LimSup is {val_limsup} for the given play: \n {rn_play.path}")
        print(f"val for payoff LimInf is {val_liminf} for the given play: \n {rn_play.path}")

    @deprecated
    def test_sup_and_liminf_limsup(self, gmax_graph: nx.MultiDiGraph, org_graph: nx.MultiDiGraph) -> None:
        # get a strategy from m = 100 for the org_graph
        eve_states, adam_states = self.get_eve_adam_states(gmax_graph)
        trail = self.strategy_synthesis_w_finite_memory(graph=gmax_graph, m=10,
                                                             _eve_states=eve_states,
                                                             _adam_states=adam_states)

        # compute imf and sup value for a play on graph say m = 10 and from vertex 1

        # get the sequence of edges for the give play
        w_max = []
        w = []
        # get key with init Flag = True
        # rn_vertex = random.choice(list(trail.keys()))
        init_vertex = str(('v1', '2'))
        rn_play = random.choice(list(trail.get(init_vertex)))
        # play = trail.get(rn_vertex)[rn_play].path
        print(rn_play.path)
        # get the corresponding play in org_graph
        g_play = self.Gmin_to_G_play(rn_play.path)
        print(g_play)
        # return
        for i in range(len(rn_play.path) - 1):
            w.append(org_graph.get_edge_data(g_play[i], g_play[i + 1])[0]['weight'])
            w_max.append(gmax_graph.get_edge_data(rn_play.path[i], rn_play.path[i + 1])[0]['weight'])
        val_sup = PayoffFunc.Sup(w)
        val_sup_max = PayoffFunc.Sup(w_max)
        # val_inf = PayoffFunc.Inf(w)
        print(f"val for payoff sup is {val_sup} for the given play: \n {g_play}")
        print(f"val for payoff inf is {val_sup_max} for the given play: \n {rn_play.path}")

        val_liminf = PayoffFunc.LimInf(w_max)
        val_limsup = PayoffFunc.LimSup(w_max)
        print(f"val for payoff LimSup is {val_limsup} for the given play: \n {rn_play.path}")
        print(f"val for payoff LimInf is {val_liminf} for the given play: \n {rn_play.path}")

def main() -> None:
    # a main routine to create a the graph and implement the strategy synthesis
    graph_obj: Graph = Graph(True)

    # create a multigraph
    org_graph = graph_obj.create_multigrpah()
    graph_obj.graph = org_graph

    if print_edge:
        graph_obj.print_edges()

    # file to store the yaml for plotting it in graphviz
    graph_obj.file_name = 'config/org_graph'

    # dump the graph to yaml
    graph_obj.dump_to_yaml()

    # read the yaml file
    graph_obj.graph_yaml = graph_obj.read_yaml_file(graph_obj.file_name)
    graph_obj.plot_fancy_graph()

    # test Gmin construction
    # construct Gmin
    Gmin = graph_obj.construct_Gmin(org_graph)
    graph_obj.graph = Gmin

    graph_obj.file_name = 'config/Gmin_graph'
    # dump to yaml file and plot it
    graph_obj.dump_to_yaml()
    graph_obj.graph_yaml = graph_obj.read_yaml_file(graph_obj.file_name)
    graph_obj.plot_fancy_graph()

    # test Gmax construction
    # construct Gmin
    Gmax = graph_obj.construct_Gmax(org_graph)
    graph_obj.graph = Gmax

    graph_obj.file_name = 'config/Gmax_graph'
    # dump to yaml file and plot it
    graph_obj.dump_to_yaml()
    graph_obj.graph_yaml = graph_obj.read_yaml_file(graph_obj.file_name)
    graph_obj.plot_fancy_graph()

if __name__ == "__main__":
    main()
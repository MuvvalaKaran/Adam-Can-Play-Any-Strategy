import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import yaml
import copy

from graphviz import Digraph


import networkx.readwrite.nx_yaml as nx_yaml
print_edge = False
debug = True

# set this boolean to true if you want to test Gmin and Gmax construction on a smaller graph
test_case = False


class Strategy(object):
    def __init__(self, path, player):
        """
        A class to hold all the strategies
        :param path:
        :type path:
        :param player:
        :type player:
        """
        self.path = path
        self.player = player
        # dictionary of form "m" : "all paths"
        self.dict = {}

    def setpath(self, newpath):
        self.path = newpath

    def setplayer(self, player):
        self.player = player

    def updatedict(self, key, value):
        self.dict.update({key: value})

class Graph(object):

    def __init__(self, save_flag=True):
        self.file_name = None
        self.graph_yaml = self.read_yaml_file(self.file_name)
        self.save_flag = save_flag
        self.graph = None

    @staticmethod
    def read_yaml_file(config_files):
        """
            reads the yaml file data and stores them in appropriate variables
        """
        graph = None

        if config_files is not None:
            with open(config_files + ".yaml", 'r') as stream:
                data_loaded = yaml.load(stream, Loader=yaml.Loader)

            graph = data_loaded['graph']

        return graph

    @staticmethod
    def plot_fancy_graph(graph):
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        :rtype:
        """
        dot = Digraph(name="graph")
        nodes = graph.graph_yaml["vertices"]
        for n in nodes:
            if n[1]['player'] == 'eve':
                if n[1].get('init'):
                    dot.attr('node', style='filled', fillcolor='red')
                else:
                    dot.attr('node', style='filled', fillcolor='lightgrey')
                dot.attr('node', shape='rectangle')
                # dot.node('eve_{}'.format(n[0]))
                dot.node(str(f"v{n[0]}"))
            else:
                if n[1].get('init'):
                    dot.attr('node', style='filled', fillcolor='red')
                else:
                    dot.attr('node', style='filled', fillcolor='lightgrey')
                dot.attr('node', shape='circle')
                # dot.node('adam_{}'.format(n[0]))
                dot.node(str(f"v{n[0]}"))

        # add all the edges
        edges = graph.graph_yaml["edges"]

        # load the weights to illustrate on the graph
        # weight = graph["weights"]
        for counter, edge in enumerate(edges):
            dot.edge(str(f"v{edge[0]}"), str(f"v{edge[1]}"), label=str(edge[2]['weight']))

        # set graph attributes
        # dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if graph.save_flag:
            graph_name = str(graph.graph.__getattribute__('name'))
            graph.save_dot_graph(dot, graph_name, True)


    def save_dot_graph(self, dot_object, graph_name, view=False):
        """
        :param dot_object: object of @Diagraph
        :type Digraph
        :param view: flag for viewing the object
        :type view: bool
        :return: a file of pdf format for now
        :rtype:
        """
        if view:
            dot_object.view(cleanup=True)

        dot_object.render(f'graph/{graph_name}', view=view, cleanup=True)

    # create a sample multigraph
    def create_multigrpah(self):
        """
        Method to create a multigraph
        :return: Multigraph constructed in networkx
        :rtype: graph
        """
        MG = nx.MultiDiGraph(name="org_graph")

        if test_case:
            MG.add_nodes_from([1, 2, 3])
            MG.add_weighted_edges_from([(1, 2, 1),
                                        (2, 1, -1),
                                        (1, 3, 1),
                                        (3, 3, 0.5)
                                        ])

            # assgin each node a player - this is then later used to plot them conveniently
            MG.nodes[1]['player'] = 'eve'
            MG.nodes[2]['player'] = 'adam'
            MG.nodes[3]['player'] = 'adam'

        else:
            MG.add_nodes_from([1, 2, 3, 4, 5])
            MG.add_weighted_edges_from([(1, 2, 1),
                                        (2, 1, -1),
                                        (1, 3, 1),
                                        (3, 3, 0.5),
                                        (3, 5, 1),
                                        (2, 4, 2),
                                        (4, 4, 2),
                                        (5, 5, 1)
                                        ])

            # assgin each node a player - this is then later used to plot them conveniently
            MG.nodes[1]['player'] = 'eve'
            MG.nodes[2]['player'] = 'adam'
            MG.nodes[3]['player'] = 'adam'
            MG.nodes[4]['player'] = 'eve'
            MG.nodes[5]['player'] = 'eve'

        # add node 1 as the initial node
        MG.nodes[1]['init'] = True

        self.graph = MG

        return MG

    def print_edges(self):
        for (u, v, wt) in self.graph.edges.data('weight'):
            print(f"({u}, {v}, {wt})")

    def dump_to_yaml(self, graph):

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
        :type graph: graph of networkx
        :param file_name: Name of the original graph yaml
        :type : str
        :return: None
        """

        data = dict(
            graph=dict(
                vertices=[node for node in graph.nodes.data()],
                edges=[edge for edge in graph.edges.data()]
            )
        )

        config_file_name = str(self.file_name + '.yaml')
        with open(config_file_name, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)


    def construct_Gmin(self, org_graph):
        """
        Method to construct a Gmin graph such that for payoff function Sup : Reg(G) = Reg(Gmin)

        V' = V x {w(e) | e belongs to the original set of Edges of G}
        V'e = {(v, n) belong to V' | v belongs to Ve(vertices that belong to Eve)}
        v'i = (vi, W) where W is the maximum weight value in G
        An edge exists ((u, n), (v, m)) iff (u, v) belong to E and m = min{n, w(u, v)}
        w'((u, n), (v, m)) = m
        :return:
        :rtype: Gmin
        """

        # create a new MG graph
        Gmin = nx.MultiDiGraph(name="Gmin_graph")

        # construct new set of states V'
        V_prime = [(v, w) for v in org_graph.nodes.data() for _, _, w in org_graph.edges.data('weight')]

        # find the maximum weight in the og graph(G)
        # specifically adding self.graph.edges.data('weight') to a create to tuple where the
        # third element is the weight value
        max_edge = max(dict(org_graph.edges).items(), key=lambda x: x[1]['weight'])
        W = max_edge[1].get('weight')

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

        if debug:
            print(Gmin.nodes.data())

        # constructing edges as per the requirement mentioned in the doc_string
        for parent in Gmin.nodes:
            for child in Gmin.nodes:
                if org_graph.has_edge(parent[0], child[0]):
                    if child[1] == min(parent[1], org_graph.get_edge_data(1, 2)[0]['weight']):
                        Gmin.add_edge(parent, child, weight=child[1])

        if debug:
            for (u, v, wt) in Gmin.edges.data('weight'):
                print(f"({u}, {v}, {wt})")

        return Gmin

    def construct_Gmax(self, org_graph):
        """
        Method to construct a Gmax graph such that for payoff function Sup : Reg(G) = Reg(Gmax)

        V' = V x {w(e) | e belongs to the original set of Edges of G}
        V'e = {(v, n) belong to V' | v belongs to Ve(vertices that belong to Eve)}
        v'i = (vi, W) where W is the maximum weight value in G
        An edge exists ((u, n), (v, m)) iff (u, v) belong to E and m = max{n, w(u, v)}
        w'((u, n), (v, m)) = m
        :return:
        :rtype: Gmax
        """

        # create a new MG graph
        Gmax = nx.MultiDiGraph(name="Gmax_graph")

        # construct new set of states V'
        V_prime = [(v, w) for v in org_graph.nodes.data() for _, _, w in org_graph.edges.data('weight')]

        # find the maximum weight in the og graph(G)
        # specifically adding self.graph.edges.data('weight') to a create to tuple where the
        # third element is the weight value
        max_edge = max(dict(org_graph.edges).items(), key=lambda x: x[1]['weight'])
        W = max_edge[1].get('weight')

        # assign nodes to Gmin with player as attributes to each node
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

        # constructing edges as per the requirement mentioned in the doc_string
        for parent in Gmax.nodes:
            for child in Gmax.nodes:
                if org_graph.has_edge(parent[0], child[0]):
                    if child[1] == max(parent[1], org_graph.get_edge_data(1, 2)[0]['weight']):
                        Gmax.add_edge(parent, child, weight=child[1])

        if debug:
            for (u, v, wt) in Gmax.edges.data('weight'):
                print(f"({u}, {v}, {wt})")

        return Gmax

    # helper method to get states that belong to eve and adam respectively
    def get_eve_adam_states(self,graph):
        """
        A method to retrieve the states that belong to eve and adam
        :param graph:
        :type graph: netowkrx
        :return: (eve_states, adam_state)
        :rtype: tuple
        """

        eve_states = []
        adam_states = []

        for n in graph.nodes.data():
            if n[1]['player'] == 'eve':
                eve_states.append(n)
            else:
                adam_states.append(n)

        return eve_states, adam_states

    # use this method to create a range of strategies
    def create_set_of_strategies(self, graph, bound):
        """
        Hypothetically G for eve and adam should be
        :param range:
        :type range:
        :return: None
        :rtype: None
        """
        # trim all non-essential stuff
        states = self.get_eve_adam_states(graph)

        # create eve and adam list to hold the vertex labels
        _eve_states = []
        _adam_states = []

        # states[0] belong to eve and state[1] belong to adam
        for e in states[0]:
            _eve_states.append(e[0])

        for a in states[1]:
            _adam_states.append(a[0])

        pathx = Strategy([], None)
        for m in range(1, bound):
            strs = self.strategy_synthesis_w_finite_memory(graph, m, _eve_states, _adam_states, pathx)

        return strs

    # use this method to create a set of strategies for a give memory (m) value
    def strategy_synthesis_w_finite_memory(self, graph, m, _eve_states, _adam_states, pathx):
        # m = 1 denotes memoryless strategy and m = n denotes you roll out n times excluding the initial vertex
        # so you rollout n + 1 times
        paths = {}


        for n in graph.nodes():
            paths.update({str(n): self.compute_all_path(graph, n, m, _eve_states, _adam_states, pathx)})
            # add this path to the str dict
            pathx.updatedict(m, paths)
        return pathx

    def compute_all_path(self, graph, curr_node, m, _eve_state, _adam_states, pathx):
        """
        A method to synthesize a m memory startegy
        :param graph: the graph on which on compute the strategy
        :type graph: Networkx
        :param m:
        :type m: int
        :return: a dictionary of strategy
        :rtype:
        """

        if type(m) is not int:
            raise ValueError

        # initially pathx is an empty list []
        path = copy.deepcopy(pathx)
        newpath = path.path + [curr_node]
        path.setpath(newpath)
        if not path.player:
            path.setplayer("eve" if path.path[0] in _eve_state else "adam")
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


def main():
    # a main routine to create a the graph and implement the strategy synthesis
    graph_obj = Graph(False)

    # create a multigraph
    org_graph = graph_obj.create_multigrpah()
    graph_obj.graph = org_graph

    if print_edge:
        graph_obj.print_edges()

    # file to store the yaml for plotting it in graphviz
    graph_obj.file_name = 'config/org_graph'

    # dump the graph to yaml
    graph_obj.dump_to_yaml(graph_obj.graph)

    # read the yaml file
    graph_obj.graph_yaml = graph_obj.read_yaml_file(graph_obj.file_name)
    graph_obj.plot_fancy_graph(graph_obj)

    # test Gmin construction
    # construct Gmin
    Gmin = graph_obj.construct_Gmin(org_graph)
    graph_obj.graph = Gmin

    graph_obj.file_name = 'config/Gmin_graph'
    # dump to yaml file and plot it
    graph_obj.dump_to_yaml(graph_obj.graph)
    graph_obj.graph_yaml = graph_obj.read_yaml_file(graph_obj.file_name)
    graph_obj.plot_fancy_graph(graph_obj)

    # test Gmax construction
    # construct Gmin
    Gmax = graph_obj.construct_Gmax(org_graph)
    graph_obj.graph = Gmax

    graph_obj.file_name = 'config/Gmax_graph'
    # dump to yaml file and plot it
    graph_obj.dump_to_yaml(graph_obj.graph)
    graph_obj.graph_yaml = graph_obj.read_yaml_file(graph_obj.file_name)
    graph_obj.plot_fancy_graph(graph_obj)

    # get eve and adam states
    eve_states, adam_states = graph_obj.get_eve_adam_states(Gmax)
    # graph_obj.strategy_synthesis_w_finite_memory(org_graph, None, 2, [])

    # trail = graph_obj.strategy_synthesis_w_finite_memory(org_graph, 3, _eve_states ,_adam_states)
    # print(trail)
    # for k, v in trail.items():
    #     print(k, [(value.path, value.player)  for value in v])
    #     print(f"for vertex {k}, the number of paths are {len(v)}")

    strs = graph_obj.create_set_of_strategies(org_graph, 5)

    for k, v in strs.dict.items():
        # print(k, [(value.path, value.player)  for value in v])
        print(f"for vertex {k}, the number of paths are {len(v)}")

if __name__ == "__main__":
    main()
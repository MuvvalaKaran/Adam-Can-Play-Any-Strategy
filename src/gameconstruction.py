import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import yaml
from graphviz import Digraph

import networkx.readwrite.nx_yaml as nx_yaml
print_edge = False
debug = True
test_case = False

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


def main():
    # a main routine to create a the graph and implement the strategy synthesis
    graph_obj = Graph(True)

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


if __name__ == "__main__":
    main()
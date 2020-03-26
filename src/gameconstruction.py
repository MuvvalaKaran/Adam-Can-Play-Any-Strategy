import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import yaml
from graphviz import Digraph

import networkx.readwrite.nx_yaml as nx_yaml

class Graph(object):

    def __init__(self, file_name):
        self.graph = Graph.read_yaml_file(file_name)

    @staticmethod
    def read_yaml_file(config_files):
        """
            reads the yaml file data and stores them in appropriate variables
        """
        with open(config_files + ".yaml", 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        graph = data_loaded['graph']

        return graph

    @staticmethod
    def create_fancy_graph(graph, save_flag):
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        :rtype:
        """
        dot = Digraph(name="Astar graph")
        nodes = graph["vertices"]
        for n in nodes:
            if n == 'g':
                dot.attr('node', shape='rectangle')
                dot.node(n, 'goal')
            elif n == 's':
                dot.node(n, 'start')
            else:
                dot.node(n, n)

        # add all the edges
        edges = graph["edges"]

        # load the weights to illustrate on the graph
        weight = graph["weights"]
        for counter, edge in enumerate(edges):
            dot.edge(edge[1], edge[3], label=str(weight[counter]))

        if save_flag:
            Graph.save_dot_graph(dot, False)
    @staticmethod
    def save_dot_graph(self, dot_object, view):
        """
        :param dot_object: object of @Diagraph
        :type Digraph
        :param view: flag for viewing the object
        :type view: bool
        :return: a file of pdf format for now
        :rtype:
        """
        if view:
            dot_object.view()
        dot_object.render('graph-q1/gv', view=view)

class figure_methods(object):

    def __init__(self, ax, fig, label, xlabel, ylabel, title, num):
        """

        :param ax: ax handle
        :type ax:
        :param fig:
        :type fig:
        """
        self.ax = ax
        self.fig = fig
        self.label = label
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.num = num

    def save_fig(self, name, dpi, bbox_param):
        """

        :param name: name of the figure with its extension
        :type name: str
        :param dpi: dots per inch
        :type dpi: as a parameter is only available when saving as png
        :param bbox_param: bbox in inches
        :type bbox_param: str ('tight') or bbox

        : e.g save_fig('__question_part_other_details.svg', dpi = None if not png/jpg/jpeg , bbox = 'tight')
        """

        self.fig.savefig(name, dpi=dpi, bbox_inches=bbox_param)

    def plotting_figs(self, x_args, y_args, color_agrs):
        """

        :param num: the figure number for reference
        :type num:
        :return: handle to the figure
        :rtype:
        """

        plt.figure(self.num)
        plt.plot(x_args, y_args, color_agrs, label=self.label)

        return plt

print_edge = True
# create a sample multigraph
def create_MG():
    MG = nx.MultiDiGraph()
    MG.add_nodes_from([1, 2, 3])
    MG.add_weighted_edges_from([(1, 2, 2),
                                (2, 1, -1),
                                (1, 3, 1),
                                (3, 3, 0.5),
                                ])
    # MG.add_edge(1, 2, weight=1)
    # MG.add_edge(2, 1, weight=-1)
    # MG.add_edge(1, 3, weight=1)

    # add a self loop at 3 and
    # MG.add_edge(3, 3, weight=0.5)

    return MG

def print_edges(graph):

    for (u, v, wt) in graph.edges.data('weight'):
        print(f"({u}, {v}, {wt})")



def plot_graph(graph):


    # dot_graph = npd.to_pydot(graph)
    # npd.write_dot(graph, 'sample.dot')
    # path = 'sample.dot'
    # G = pgv.AGraph(path)
    # # G.layout(prog='dot')
    # G.draw('file.pdf', prog='dot')



    # G = pgv.AGraph(dot_graph)
    # G.draw('file.png')
    # create a graph viz grpah object to plot our multidirected graph
    # G = pgv.AGraph(strict=False, directed=True)
    #
    # # create a node list
    # node_list = list(graph.nodes)
    # G.add_nodes_from(node_list)
    #
    # for _,_,wt in graph.edges.data('weight'):

    # pos = nx.nx_agraph.graphviz_layout(graph)
    # nx.draw(graph, pos=pos)
    # write_dot(graph, 'file.dot')

    # sample to test pygraphviz is working or not
    d = {'1': {'2': None}, '2': {'1': None, '3': None}, '3': {'2': None}}
    G = pgv.AGraph(d)
    G.draw('file.png', prog='dot')


def dump_to_yaml(graph):

    # given the graph dump the content to yaml file
    # nx_yaml.write_yaml(graph, 'sample.yaml')
    # sample to dump

    document = """"
    graph:
        vertices:
         - (s, 1)
         - (a, 1)
         - (b, 2)
        edges:
         - ()
         - ()

    """

    data = dict(
        graph=dict(
            vertices = [('s',1), ('a',1), ('b', 2)],
            edges = [(), (), ()]
        )
    )

    with open('sample.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)



def main():
    # a main rotuine to create a the graph and implement the strategy synthesis
    graph = create_MG()
    if print_edge:
        print_edges(graph)

    print(dict(graph.edges))
    dump_to_yaml(graph)
    # plot_graph(graph)
    # plt.plot(111)
    # nx.draw(graph, with_labels=True)
    # plt.show()

if __name__=="__main__":
    main()
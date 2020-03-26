import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import yaml
from graphviz import Digraph

import networkx.readwrite.nx_yaml as nx_yaml

class Graph(object):

    def __init__(self, file_name, save_flag=True):
        self.graph = self.read_yaml_file(file_name)
        self.save_flag = save_flag

    def read_yaml_file(self, config_files):
        """
            reads the yaml file data and stores them in appropriate variables
        """
        with open(config_files + ".yaml", 'r') as stream:
            data_loaded = yaml.load(stream, Loader=yaml.Loader)

        graph = data_loaded['graph']

        return graph

    def create_fancy_graph(self):
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        :rtype:
        """
        dot = Digraph(name="graph")
        nodes = self.graph["vertices"]
        for n in nodes:
            if n[1]['player'] == 'eve':
                dot.attr('node', shape='rectangle')
                # dot.node('eve_{}'.format(n[0]))
                dot.node(str(f"v{n[0]}"))
            else:
                dot.attr('node', shape='circle')
                # dot.node('adam_{}'.format(n[0]))
                dot.node(str(f"v{n[0]}"))

        # add all the edges
        edges = self.graph["edges"]

        # load the weights to illustrate on the graph
        # weight = graph["weights"]
        for counter, edge in enumerate(edges):
            dot.edge(str(f"v{edge[0]}"), str(f"v{edge[1]}"), label=str(edge[2]['weight']))

        # set graph attributes
        dot.graph_attr['rankdir'] = 'LR'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1')

        if self.save_flag:
            self.save_dot_graph(dot, True)


    def save_dot_graph(self, dot_object, view=False):
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
        dot_object.render('graph/og_graph', view=view, cleanup=True)

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
    MG.add_weighted_edges_from([(1, 2, 1),
                                (2, 1, -1),
                                (1, 3, 1),
                                (3, 3, 0.5),
                                ])

    # assgin each node a player - this is then later used to plot them conveniently
    MG.nodes[1]['player'] = 'eve'
    MG.nodes[2]['player'] = 'adam'
    MG.nodes[3]['player'] = 'adam'

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


def dump_to_yaml(graph, file_name):

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

    config_file_name = str(file_name + '.yaml')
    with open(config_file_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)



def main():
    # a main rotuine to create a the graph and implement the strategy synthesis
    graph = create_MG()
    if print_edge:
        print_edges(graph)

    print(dict(graph.edges))
    file_name = 'config/org_graph'
    # dump content to yaml config file
    dump_to_yaml(graph, file_name)

    # call the graph class to create a graph
    g = Graph(file_name, True)
    g.create_fancy_graph()

    # plot_graph(graph)
    # plt.plot(111)
    # nx.draw(graph, with_labels=True)
    # plt.show()

if __name__=="__main__":
    main()
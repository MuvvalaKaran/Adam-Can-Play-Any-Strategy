import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import yaml
import copy
import random

from graphviz import Digraph
from src.PayoffFunc import PayoffFunc
# import sys
# print(sys.path)

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
test_case = False

# test inf payoff function
test_inf = False
# test sup payoff function
test_sup = True


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
        self.init = None
        # dictionary of form "m" : "all paths"
        # self.dict = {}

    def setinit(self, init):
        '''
        Init flag set to be True for strategies begining from initial vertex
        :param init: Boolean variable idicating strategies starting from the init vertex
        :type init: bool
        :return: None
        :rtype: None
        '''
        self.init = init

    def setpath(self, newpath):
        self.path = newpath

    def setplayer(self, player):
        self.player = player

    # def updatedict(self, key, value):
    #     self.dict.update({key: value})

class Graph(object):

    def __init__(self, save_flag=True):
        self.file_name = None
        self.graph_yaml = self.read_yaml_file(self.file_name)
        self.save_flag = save_flag
        self.graph = None
        self.Strs_dict = {}

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
                                        (2, 1, 2),
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

        if print_Gmin_nodes:
            print("Printing Gmin nodes : \n", Gmin.nodes.data())

        # constructing edges as per the requirement mentioned in the doc_string
        for parent in Gmin.nodes:
            for child in Gmin.nodes:
                if org_graph.has_edge(parent[0], child[0]):
                    if child[1] == min(parent[1], org_graph.get_edge_data(parent[0], child[0])[0]['weight']):
                        Gmin.add_edge(parent, child, weight=child[1])

        if print_Gmin_edges:
            print("Printing Gmin edges and with weights \n")
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
                    if child[1] == max(parent[1], org_graph.get_edge_data(parent[0], child[0])[0]['weight']):
                        Gmax.add_edge(parent, child, weight=child[1])

        if print_Gmax_edges:
            print("Printing Gmax edges and with weights \n")
            for (u, v, wt) in Gmax.edges.data('weight'):
                print(f"({u}, {v}, {wt})")

        return Gmax

    # helper method to get states that belong to eve and adam respectively
    def get_eve_adam_states(self, graph):
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
        Hypothetically G for eve and adam should be infinte. But technically we done have infinte memory to compute
        strategies with infinite memory. Also we implement recursion to compute all possible paths. The max depth of
        recursion in python is (~1000). So m < 1000 is a must.

        This method is used to compute strategies for a given range of m.
        :param graph: Graph from which we would like to compute strategies for a range of m values
        :type graph: @Networkx
        :param bound: Upper Bound on the memory
        :type bound: int
        :return: a dictionary of form {{m:set of strategies from each vertex }}
        :rtype: dict
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

        for m in range(1, bound):
            strs = self.strategy_synthesis_w_finite_memory(graph, m, _eve_states, _adam_states)
            self.Strs_dict.update({m: strs})

        return self.Strs_dict

    def strategy_synthesis_w_finite_memory(self, graph, m, _eve_states, _adam_states):
        """
        A method to compute a set of strategies for a given graph. This method calls @compute_all_paths() to compute
        all possible paths from a give vertex. While doing so each path is assigned which player that strategy belongs
        to. In this method m = 1 denotes memoryless strategy and m = n denotes you roll out n times excluding the
        initial vertex. In total the path length is n + 1 as we include the initial vertex as well.

        We can employ this method to create a set of strategies for a given memory(m) value
        :param graph: Graph from which we would like to compute strategies
        :type graph: @Networkx
        :param m:memory of the strategy
        :type m:int
        :param _eve_states:
        :type _eve_states: list
        :param _adam_states:
        :type _adam_states: lisy
        :return:a dictonary of all paths computes from each states with memory m {{vertex_label: paths} }
        :rtype: dict
        """
        paths = {}

        for n in graph.nodes():
            paths.update({str(n): self.compute_all_path(graph, n, m,
                                                        _eve_states,
                                                        _adam_states,
                                                        pathx=Strategy([], None))})
        return paths

    def compute_all_path(self, graph, curr_node, m, _eve_state, _adam_states, pathx):
        """
        A method to compute all the paths possible from a given state. This function is called recursively until
        memory(m) becaome 0. So, technically we rollout m + 1 times. with the first vertex in the path being the vertex
        from where we begin compute paths
        :param graph: Graph from which we would like to compute all possible path from a vertex
        :type graph: @Networkx
        :param curr_node: Current node
        :type curr_node: graph.node
        :param m: memory
        :type m: int
        :param _eve_state: set of states that belong to eve
        :type _eve_state: list
        :param _adam_states: set of states that belong to adam
        :type _adam_states: list
        :param pathx: path that keeps getting appended with every recursion
        :type pathx: @Strategy
        :return: a list of all paths from the given vertex. It includes self-loops as well
        :rtype: list
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
            if path.path[0] == 1 or path.path[0] == (1,2):
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

    def get_set_of_strategies(self):
        return self.Strs_dict

    def print_set_of_strategies(self):
        for k, v in self.Strs_dict.items():
            print(f"For memory {k} :")
            for vertex, pths in v.items():
                print(f"for vertex {vertex}, the number of paths is {len(pths)}")
            print("")

    # helper method to get the corresponding play from Gmin/Gmax to G
    def Gmin_to_G_play(self, play):
        # all the leading vertex values belong to the org_graph
        # here play is a sequence of tuple
        G_play = []

        for i in play:
            G_play.append(i[0])

        return G_play

    def test_inf_and_liminf_limsup(self, gmin_graph, org_graph):
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
        init_vertex = str((1, 2))
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

    def test_sup_and_liminf_limsup(self, gmax_graph, org_graph):
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
        init_vertex = str((1, 2))
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

    # if you want to get a particular strategy with memory say m = 10 for a graph
    if print_str_m:
        eve_states, adam_states = graph_obj.get_eve_adam_states(Gmax)
        trail = graph_obj.strategy_synthesis_w_finite_memory(graph=Gmin, m=10,
                                                     _eve_states=eve_states,
                                                     _adam_states=adam_states)
        for k, v in trail.items():
            print(k, [(value.path, value.player)  for value in v])
            print(f"for vertex {k}, the number of paths are {len(v)}")

    graph_obj.create_set_of_strategies(Gmin, 10)

    if print_range_str_m:
        graph_obj.print_set_of_strategies()

    if test_inf:
        graph_obj.test_inf_and_liminf_limsup(Gmin, org_graph)
    if test_sup:
        graph_obj.test_sup_and_liminf_limsup(Gmax, org_graph)



if __name__ == "__main__":
    main()
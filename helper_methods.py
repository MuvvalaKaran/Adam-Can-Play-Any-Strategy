# A decorator to thow warning when we use deprecated methods/functions/routines
import warnings
import re
import subprocess as sp
import math

from typing import Dict, Tuple
from bidict import bidict

# import local packages
# from src.graph import Graph
# value to replcae with if you have inf/-inf as edge weight in g_hat : can happen for cumulative payoff
MAX_CONST = -1000000

# absolute addresses of mpg solver toolbox and where to read the g_hat.mpg file and dump the strategy in
CONFIG_DIR = "mpg/"
MPG_ABS_DIR = "/home/karan-m/Documents/mean_payoff_games/gpumpg/bin/"
MPG_OP_ABS_DIR = "/home/karan-m/Documents/Research/variant_1/Adam-Can-Play-Any-Strategy/mpg/"


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""
    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc


def create_mpg_file(graph, name: str) -> Tuple[bidict, list]:
    """
    A method to create a file which adheres to the input file
    format used by the mean payoff games toolbox.

    We then use this file as input the payoff toolbox to compute
    the antagonistic value for that game.

    The format of the file dumped is
    <NODE> <MIN/MAX> <ADJ_NODE>:<EDGE_WEIGHT>...; # for all the nodes in @graph

    The nodes over here are internally mapped to a unique number and then dumped to the file.
    :param graph: The graph to be dumped in the respective format
    :param name: The name of the file
    :return:
    """
    # if not isinstance(graph, Graph):
    #     warnings.warn(f"Please make sure that the graph is of type Graph")

    directory_add: str = CONFIG_DIR + name + ".mpg"
    # map is of the format : {state_name: str : hashed_int_val : int}
    _node_index_map = bidict({state: index
                              for index, state
                              in enumerate(graph._graph.nodes)})

    f = open(directory_add, "w")

    # get all the initial nodes of g_b(s)
    g_b_init_nodes = [_s for _s in graph._graph.successors("v1")]

    # start dumping
    for node in graph._graph.nodes():
        if graph._graph.nodes[node]["player"] == "adam":
            player = "MIN"
        else:
            player = "MAX"

        # create the line
        f.write(f"{_node_index_map[node]} {player} ")

        # get outgoing edges of a graph
        for ie, e in enumerate(graph._graph.edges(node)):
            mapped_next_node = _node_index_map[e[1]]
            edge_weight = graph._graph[e[0]][e[1]][0]['weight']

            if edge_weight == math.inf or edge_weight == -1 * math.inf:
                edge_weight = MAX_CONST

            if ie == len(graph._graph.edges(node)) - 1:
                f.write(f"{mapped_next_node}:{int(edge_weight)}; \n")

            else:
                f.write(f"{mapped_next_node}:{int(edge_weight)}, ")

    return _node_index_map, g_b_init_nodes


def read_mpg_op(_node_index_map: bidict, file_name: str, _g_b_init_nodes: list, debug : bool = False):
    """
    A method that reads the output of the mean payoff game tool and returns a dictionary

    The dictionary mapping from each to the next state

    :param name: The name of the file to read
    :return:
    """
    # get the number associated with 'v1'
    # NODE_V1_IDX: int = _node_index_map.inverse["v1"]

    # dictionary to hold reg values
    reg_dict: Dict[int, float] = {}

    f = open(file_name, "r")
    str_dict = {}

    for line in f.readlines():
        curr_node, _, next_node, mean_val = line.split()

        str_dict.update({_node_index_map.inverse[int(curr_node)]:
                         _node_index_map.inverse[int(next_node)]})

        reg_dict.update({curr_node: mean_val})

    b_val = str_dict["v1"][1]
    print("******************printing Reg value********************")
    print(f"Playing in graph g_b = {b_val}")
    for _n in _g_b_init_nodes:
        print(f"Reg value from node {_n} is -1 * {reg_dict[str(_node_index_map[_n])]}")

    if debug:
        for curr_n, next_n in str_dict.items():
            print(f"The current node is {curr_n} and the strategy is {next_n}")

    return str_dict


def run_save_output_mpg(graph, name: str, go_fast: bool = True, debug: bool = False) -> Dict:
    """
    A helper method to run the mpg solver through terminal given a graph.

    First we dump the grpah in a format readable by mpg, interpret the output and only
    save final output of the game and then map the hashed nodes to states nodes on graph
    and return that dictionary.
    :param graph: The graph on which we would like compute the antagonistic value
    :param name: The name of file to dump the graph in. The name will be of format <name>.mpg
                 NOTE: The output of mpgsolver toolbox will be saved as <name>_str.txt
    :return: The final str dictionary
    """
    print("**************************Computing Reg Minimizing Strategy on G_hat*************************")
    # dump the graph
    _node_index_map, _g_b_init_nodes = create_mpg_file(graph, name)

    ip_file_name = CONFIG_DIR + name + ".mpg"
    op_file_name = MPG_OP_ABS_DIR + f"{name}_str.txt"
    mpg_dir_call = MPG_ABS_DIR + "mpgsolver"

    mpgsolver_call = [mpg_dir_call, "-d 2", ip_file_name, " > ", op_file_name]

    if go_fast:
        with open(op_file_name, 'w') as fh:
            completed_process = sp.run(mpgsolver_call,
                                       stdout=fh, stderr=sp.PIPE)
            _curate_op_data(op_file_name)
    else:
        completed_process = sp.run(mpgsolver_call,
                                   stdout=sp.PIPE, stderr=sp.PIPE)

    if not go_fast:
        call_string = completed_process.stdout.decode()
        print('%s' % call_string)

    str_dict = read_mpg_op(_node_index_map, op_file_name, _g_b_init_nodes, debug=debug)

    return str_dict


def _read_mpg_file(file_name: str):
    """
    A helper method to read the output dumped by mpgsolver from the file @filename
    :param file_name: The abs path of the file
    :return:
    """

    try:
        with open(file_name) as fh:
            return fh.readlines()

    except FileNotFoundError:
        print(f"The file {file_name} does not exists")


def _curate_op_data(file_name):
    """
    A helper method to trim away all the unnecessary data for read_mpg_op to
     read the final strategy and and map it back to the graph (in our case g_hat)
    :param data:
    :return:
    """

    start_re_flag = 'Starting writing solution'
    stop_re_flag = 'Finished writing solution'

    lines = _read_mpg_file(file_name)

    start_index = [index for index, line in enumerate(lines) if re.search(start_re_flag, line)][0]
    stop_index = [index for index, line in enumerate(lines) if re.search(stop_re_flag, line)][0]

    with open(file_name, "w") as fh:
        fh.write("".join(lines[start_index + 1: stop_index]))

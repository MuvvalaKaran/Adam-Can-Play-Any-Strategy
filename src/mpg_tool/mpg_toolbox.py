# a file that contains all the methods relevant to running the mean payoff games toolbox and interpret the output
import warnings
import re
import subprocess as sp
import math
import warnings
import sys

from ..config import MPG_ABS_DIR, MPG_OP_ABS_DIR
from typing import Dict, Tuple, Optional
from bidict import bidict
from ..graph import TwoPlayerGraph

# value to replcae with if you have inf/-inf as edge weight in g_hat : can happen for cumulative payoff
MAX_CONST = -1000000

# absolute addresses of mpg solver toolbox and where to read the g_hat.mpg file and dump the strategy in
CONFIG_DIR = "mpg/"


class MpgToolBox:
    """
    A class with relevant method call to the mpg toolbox
    graph : The input graph to the mpg toolbox
    file_name : The name of the input file (without any extension) for the graph to be dumped in
    node_index_map : An internal bi-directional mapping that maps an arbitrary node as tuple to an int representation.
    Methods:
        1. create_mpg_file : A method to dump a given graph into an appropriate format that the mpg toolbox recognizes
        2. compute_cVal : A method to compute the strategy for a cooperative game
        3. compute_reg_val : A method to compute the strategy for a competitive game
    Usage of toolbox:
    Usage: ./mpgsolver [OPTIONS] FILES
    Solve the mean-payoff game in FILES on the GPU.
    Options:
      -h [ --help ]                  display this help text and exit
      -i [ --input-file-format ] arg input file format: mpg, pg, hoa or bin
      -o [ --output-file ] arg       output file
      -c [ --cycle-check ]           explicitly check for non-positive cycles in
                                     the game graph
      --scc-partition                compute the scc partition of the game arena
      --no-compact-colors            do not compact the colors by removing unused
                                     colors
      --no-auxilliary-nodes          do not add auxilliary nodes to the arena (may
                                     lead to wrong results)
      --sort-by-owner                sort nodes by owner (within scc with scc
                                     partitions)
      -d [ --device ] arg            select the device (by index or name)
      -w [ --work-group-size ] arg   set the work group size (0=auto)
      -p [ --print-arena ]           print the arena
      --print-arena-info             print basic arena information
      --device-info                  print device information
      --profiling                    measure and print kernel profiling information
      -t [ --timing ]                measure and print timing information
      --no-solving                   do not solve the game
      --zero-mean-partition          only solve the zero mean partition problem
      --check-realizability          check realizability
    """
    def __init__(self, graph: TwoPlayerGraph, file_name: str = None):
        self._mpg_graph = graph
        self._file_name: Optional[str] = file_name
        self.node_index_map: Optional[bidict] = None

    @property
    def mpg_graph(self):
        return self._mpg_graph

    @property
    def node_index_map(self):
        return self._node_index_map

    @property
    def file_name(self):
        return self._file_name

    @mpg_graph.setter
    def mpg_graph(self, mpg_graph):
        if mpg_graph is None:
            warnings.warn("Please make sure that the graph is not Empty")

        if not isinstance(mpg_graph, TwoPlayerGraph):
            warnings.warn(f"Please ensure that the graph is of type of TwoPlayerGraph."
                          f"Currently it is of type {type(mpg_graph)}")

        if len(mpg_graph._graph.nodes()) == 0:
            warnings.warn(f"Looks like the graph has no nodes at all!")

        self._mpg_graph = mpg_graph

    @file_name.setter
    def file_name(self, file_name: str):

        if not isinstance(file_name, str):
            warnings.warn(f"Please ensure that the name of the file to be dumped is a str."
                          f"Currently it is of type {type(file_name)}")

        self._file_name = file_name

    @node_index_map.setter
    def node_index_map(self, mapping: Dict):

        self._node_index_map = mapping

    def create_mpg_file(self, reg: bool = False, cval: bool  = False) -> Tuple[bidict, list]:

        """
        A method to create a file which adheres to the input file
        format used by the mean payoff games toolbox.
        We then use this file as input the payoff toolbox to compute
        the antagonistic value for that game.
        The format of the file dumped is
        <NODE> <MIN/MAX> <ADJ_NODE>:<EDGE_WEIGHT>...; # for all the nodes in @graph
        The nodes over here are internally mapped to a unique number and then dumped to the file.
        :param graph: The graph to be dumped in the respective format - should be total
        :param name: The name of the file
        :return:
        """
        directory_add: str = CONFIG_DIR + self.file_name + ".mpg"
        # map is of the format : {state_name: str : hashed_int_val : int}
        self.node_index_map = bidict({state: index
                                  for index, state
                                  in enumerate(self.mpg_graph._graph.nodes)})

        f = open(directory_add, "w")

        # start dumping
        for node in self.mpg_graph._graph.nodes():
            if reg:
                player = self._get_reg_players(node)
            elif cval:
                player = self._get_cval_players()
            else:
                warnings.warn("Please ensure either of the reg or cval flag is True")
                sys.exit(-1)

            # create the line
            f.write(f"{self.node_index_map[node]} {player} ")

            # get outgoing edges of a graph
            for ie, e in enumerate(self.mpg_graph._graph.edges(node)):
                mapped_next_node = self.node_index_map[e[1]]
                edge_weight = self.mpg_graph._graph[e[0]][e[1]][0]['weight']

                if edge_weight == math.inf or edge_weight == -1 * math.inf:
                    edge_weight = MAX_CONST

                if ie == len(self.mpg_graph._graph.edges(node)) - 1:
                    f.write(f"{mapped_next_node}:{int(edge_weight)}; \n")

                else:
                    f.write(f"{mapped_next_node}:{int(edge_weight)}, ")

        f.close()

    def _get_reg_players(self, node):
        """
        A helper method that return if its a MIN or a MAX based on the player attribute associated with the node
        :param node:
        :return:
        """
        if self.mpg_graph._graph.nodes[node]["player"] == "adam":
            return "MIN"
        else:
            return "MAX"

    def _get_cval_players(self):
        return "MAX"

    def compute_cval(self,  go_fast: bool = True, debug: bool = False):
        print("**************************Computing cVal of all nodes in G*************************")
        # dump the graph
        self.create_mpg_file(cval=True)

        ip_file_name = CONFIG_DIR + self.file_name + ".mpg"
        op_file_name = MPG_OP_ABS_DIR + f"{self.file_name}_dict.txt"
        mpg_dir_call = MPG_ABS_DIR + "mpgsolver"

        mpgsolver_call = [mpg_dir_call, "-d 2", ip_file_name, " > ", op_file_name]

        if go_fast:
            with open(op_file_name, 'w') as fh:
                completed_process = sp.run(mpgsolver_call,
                                           stdout=fh, stderr=sp.PIPE)
                self._curate_op_data(op_file_name)
        else:
            completed_process = sp.run(mpgsolver_call,
                                       stdout=sp.PIPE, stderr=sp.PIPE)

        if not go_fast:
            call_string = completed_process.stdout.decode()
            print('%s' % call_string)

        coop_dict = self.read_cval_mpg_op(op_file_name, debug)

        return coop_dict

    def compute_reg_val(self, go_fast: bool = True, debug: bool = False) -> Tuple[Dict, float]:
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
        self.create_mpg_file(reg=True)

        ip_file_name = CONFIG_DIR + self.file_name + ".mpg"
        op_file_name = MPG_OP_ABS_DIR + f"{self.file_name}_str.txt"
        mpg_dir_call = MPG_ABS_DIR + "mpgsolver"

        mpgsolver_call = [mpg_dir_call, "-d 2", ip_file_name, " > ", op_file_name]

        if go_fast:
            with open(op_file_name, 'w') as fh:
                completed_process = sp.run(mpgsolver_call,
                                           stdout=fh, stderr=sp.PIPE)
                self._curate_op_data(op_file_name)
        else:
            completed_process = sp.run(mpgsolver_call,
                                       stdout=sp.PIPE, stderr=sp.PIPE)

        if not go_fast:
            call_string = completed_process.stdout.decode()
            print('%s' % call_string)

        str_dict, final_reg_value = self.read_reg_mpg_op(op_file_name, debug=debug)

        return str_dict, final_reg_value

    def _read_mpg_file(self, file_name: str):
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

    def _curate_op_data(self, file_name):
        """
        A helper method to trim away all the unnecessary data for read_mpg_op to
         read the final strategy and and map it back to the graph (in our case g_hat)
        :param data:
        :return:
        """

        start_re_flag = 'Starting writing solution'
        stop_re_flag = 'Finished writing solution'

        lines = self._read_mpg_file(file_name)

        start_index = [index for index, line in enumerate(lines) if re.search(start_re_flag, line)][0]
        stop_index = [index for index, line in enumerate(lines) if re.search(stop_re_flag, line)][0]

        with open(file_name, "w") as fh:
            fh.write("".join(lines[start_index + 1: stop_index]))

    def read_cval_mpg_op(self, file_name: str, debug: bool = False):
        """
        A method that reads the output of the mean payoff game tool and returns a dictionary
        The dictionary is a mapping from each state to the next state
        :param file_name: The name of the file to read
        :return:
        """

        # dictionary to hold reg values
        coop_dict: Dict[tuple, float] = {}

        f = open(file_name, "r")

        for line in f.readlines():
            curr_node, _, next_node, c_val = line.split()

            try:
                c_val = eval(c_val)
            except ZeroDivisionError:
                warnings.warn(f"The c_val of node {curr_node} is {c_val}."
                              f" Cannot divide by zero. Please check the output")
                sys.exit(-1)

            coop_dict.update({self.node_index_map.inverse[int(curr_node)]: c_val})

        if debug:
            for _n, _v in coop_dict.items():
                print(f"The cVal for node : {_n} is : {_v}. \n")

        return coop_dict

    def read_reg_mpg_op(self, file_name: str, debug: bool = False):
        """
        A method that reads the output of the mean payoff game tool and returns a dictionary
        The dictionary is a mapping from each state to the next state
        :param file_name: The name of the file to read
        :return:
        """
        _g_b_init_nodes = [_s for _s in self.mpg_graph._graph.successors("v1")]

        # dictionary to hold reg values
        reg_dict: Dict[int, float] = {}

        f = open(file_name, "r")
        str_dict = {}

        for line in f.readlines():
            curr_node, _, next_node, mean_val = line.split()

            str_dict.update({self.node_index_map.inverse[int(curr_node)]:
                                 self.node_index_map.inverse[int(next_node)]})

            try:
                mean_val = eval(mean_val)
            except ZeroDivisionError:
                warnings.warn(f"The c_val of node {curr_node} is {mean_val}."
                              f" Cannot divide by zero. Please check the output")
                sys.exit(-1)

            reg_dict.update({curr_node: mean_val})

        b_val = str_dict["v1"][1]
        print("******************printing Reg value********************")
        print(f"Playing in graph g_b = {b_val}")
        for _n in _g_b_init_nodes:
            print(f"Reg value from node {_n} is: {-1 * reg_dict[str(self.node_index_map[_n])]}")

        _next_node_from_v1: tuple = str_dict["v1"]
        _next_node_hash_value: str = str(self.node_index_map[_next_node_from_v1])
        final_reg_val: float = reg_dict[_next_node_hash_value]

        if debug:
            for curr_n, next_n in str_dict.items():
                print(f"The current node is {curr_n} and the strategy is {next_n}")

        return str_dict, final_reg_val
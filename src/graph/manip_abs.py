# a file to construct the manipulation domain
from typing import List, Tuple, Dict
from collections import defaultdict

from src.graph.graph import FiniteTransSys

class Node:
    def __init__(self):
        self._no_of_objects: int = 0
        self._loc_of_interest: List[str] = []
        self._obj_locs = []
        self._gripper_locs = []
        self.obj_list = []

    def get_no_of_obj(self):
        return self._no_of_objects

    def get_locations_of_interests(self):
        return self._loc_of_interest

    def get_obj_lst(self):
        return self.obj_list

    def set_locations_of_interest(self, all_possible_locs: List[str]) -> None:
        self._loc_of_interest = all_possible_locs

    def set_no_of_obj(self, obj_count: int) -> None:
        self._no_of_objects = obj_count
        self.set_obj_names()

    def set_obj_names(self):

        assert self._no_of_objects != 0, "WARNING: trying to create objects but there are no objects to create"

        for i in range(self._no_of_objects):
            self.obj_list.append(f"cup_{i}")

    def set_obj_locs(self):
        self._obj_locs = self._loc_of_interest + ["gripper"]

    def set_gripper_locs(self):

        assert self._no_of_objects != 0, "WARNING : no objects in the space"
        assert len(self._loc_of_interest) != 0, "WARNING : no locations of interest"

        self._gripper_locs = self._loc_of_interest + self.obj_list + ["free"]


class SysActions:

    def __init__(self, base_actions: List[str]):
        self.__base_actions = base_actions

    def get_base_actions(self):
        return self.__base_actions

    def create_parameterized_action(self, u: tuple, v: tuple):
        raise NotImplementedError

    def check_drop_action(self, u: tuple, v: tuple, node_handle) -> bool:
        # check if there exists a drop edge between u and v
        if ("gripper" in u[:-1]) and (u[-1] in node_handle._loc_of_interest):
            # get which obj was being manipulated
            _idx = list(u[:-1]).index("gripper")
            obj_manip = u[_idx]
            # get the location where the gripper was going to place the object
            _loc = u[-1]
            if (v[_idx] == _loc) and (v[-1] == "free"):
                return True
        return False

    def check_grab_action(self, u: tuple, v: tuple, node_handle) -> bool:
        if (u[-1] == "free") and ("gripper" not in u[:-1]):
            # if "gripper" in v[:-1]:
            #     # get the object being grabbed
            #     _idx = list(v[:-1]).index("gripper")
            #     obj = f"cup_{_idx}"
            #     if v[-1] == obj:
            #         return True
            if (v[-1] in node_handle.obj_list) and (u[:-1] == v[:-1]):
                return True
        return False

    def check_transfer_action(self, u: tuple, v: tuple, node_handle) -> bool:
        if ("gripper" not in u[:-1]) and (u[-1] in node_handle.obj_list):
            # # self loop
            # if u == v:
            #     return True
            # get the obj ide in v node
            obj = u[-1]
            _idx = int(u[-1][-1])
            # everything else should remain the same as well
            rem_set = set([i for i in range(3)]) - {_idx}
            if (v[_idx] == "gripper") and (v[-1] in node_handle._loc_of_interest):
                for idx in rem_set:
                    if v[idx] != u[idx]:
                        return False
                    return True
        return False


    def check_transit_action(self, u: tuple, v: tuple, node_handle) -> bool:
        if ("gripper" not in u[:-1]) and (u[-1] == "free"):
            if u == v:
                return True

            if (v[-1] in node_handle.obj_list) and (v[:-1] == u[:-1]):
                return True

        return False


class Predicates:

    def __init__(self, predicates: List[str]):
        self.__atomic_propositions = predicates

    def get_predicates(self):
        return self.__atomic_propositions


class ManipAbs(FiniteTransSys):

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False):
        self._graph_name = graph_name
        self._config_yaml = config_yaml
        self._save_flag = save_flag

    # def dump_to_yaml(self) -> None:
    #     # overriding this method to suit the format that nicks wants it in
    #     raise NotImplementedError


class ManipAbsBuilder:

    def construct_manip_abs(self, object_count: int, locations: List[str],
                            sys_actions: List[str] = ["grasp", "hold", "move", "place"],
                            initial_node: tuple = ("rightbase", "elsewhere_1", "leftbase", "free"),
                            debug: bool = False, plot: bool = False):
        manip_graph = ManipAbs("manip_abs", "config/manip_abs", save_flag=True)
        manip_graph.construct_graph()

        node_handle = Node()
        node_handle.set_locations_of_interest(locations)
        node_handle.set_no_of_obj(object_count)
        node_handle.set_gripper_locs()
        node_handle.set_obj_locs()
        node_list = self.construct_nodes(node_handle, debug=debug)

        loop_up_table = self._loop_up_table(node_list)

        # add nodes to the graph
        for node in node_list:
            manip_graph.add_state(loop_up_table[node])

        # add the initial node
        manip_graph.add_initial_state(loop_up_table[initial_node])

        # add edges and their respective edge labels
        self.add_edges(manip_graph, node_list, sys_actions, node_handle, loop_up_table, debug=debug)

        if plot:
            print("*************Plotting graph***************")
            manip_graph.plot_graph()

    def _loop_up_table(self, node_list: List) -> Dict:
        # assign each node a unique number
        node_loop_up = defaultdict(lambda : -1)

        for inx, _n in enumerate(node_list):
            node_loop_up[_n] = inx

        return node_loop_up

    def construct_nodes(self, node_handle: Node, debug: bool = False) -> List[tuple]:
        print("***********************Constructing set of Nodes***********************")
        # construct all the nodes

        node_list = [(c0, c1, c2, g) for c0 in node_handle._obj_locs
                     for c1 in node_handle._obj_locs
                     for c2 in node_handle._obj_locs
                     for g in node_handle._gripper_locs]

        if debug:
            print("Printing the list of node")
            print("**************************")
            for node in node_list:
                print(node)
            print("**************************")
            print(f"Total number of nodes in the graph without sanity check is {len(node_list)}")
            print("**************************")

        # perform sanity check
        node_list = self.sanity_check(node_list, node_handle, debug=debug)

        return node_list

    def add_edges(self, abs_graph, node_list, sys_actions: List[str], node_handle, map, debug: bool = False):
        edge_count: int = 0
        sys_action = SysActions(sys_actions)
        for u_node in node_list:
            for v_node in node_list:

                # add function here that check for conditions of grab, drop, transfer and transit
                if sys_action.check_drop_action(u_node, v_node, node_handle):
                    edge_count += 1
                    abs_graph.add_edge(map[u_node], map[v_node])

                if sys_action.check_grab_action(u_node, v_node, node_handle):
                    edge_count += 1
                    abs_graph.add_edge(map[u_node], map[v_node])

                if sys_action.check_transfer_action(u_node, v_node, node_handle):
                    edge_count += 1
                    abs_graph.add_edge(map[u_node], map[v_node])
                #
                # if sys_action.check_transit_action(u_node, v_node, node_handle):
                #     edge_count += 1
                #     abs_graph.add_edge(map[u_node], map[v_node])

        if debug:
            print(f"Total Number of edges are : {edge_count}")

    def sanity_check(self, node_list, node_handle, debug: bool = False) -> List[Tuple]:
        # prune the nodes from the list that are not physically possible
        """
        Physical constraints:
        1. No two objects will have the same locations
        2. No more than one object will be in the gripper's hand |L(cupi) == gripper| = 1
        3. if L(cupi) = gripper => L(G) != "free" or L(G) != cup_i
        4. while dropping an object, the location should be free i.e if L(G) == {a location_of_interest - l}
        => for all i L(cupi) != l
        5. L(G) == {locations_of_interest - l} => L(cupi) = "gripper"
        NOTE : I do not add the physical constraint that states that an object should be in base location before
         being at the top. This should captured by the automaton.
        :return: A set of valid nodes that satisfy the above constraints
        """
        print("***********************Performing sanity check***********************")

        def constr_1(node: tuple):
            # enforces constraint # 1., 2., and 4.
            for _l in node_handle._obj_locs:
                if list(node).count(_l) >= 2:
                    return True
            return False

        def constr_2(node: tuple):
            # enforces constraint # 3
            if "gripper" in node[:-1]:
                if (node[-1] == "free") or (node[-1] in node_handle.get_obj_lst()):
                    return True
            return False

        def constr_3(node: tuple):
            # enforce # 4
            if node[-1] in node_handle._loc_of_interest:
                if "gripper" not in node[:-1]:
                    return True
            return False

        node_set = set(node_list)
        for node in node_list:
            if constr_1(node):
                node_set = node_set - {node}
                continue

            if constr_2(node):
                node_set = node_set - {node}
                continue

            if constr_3(node):
                node_set = node_set - {node}
                continue

        if debug:
            print("**************************")
            for node in node_set:
                print(node)
            print(f"No. of valid nodes after sanity check is: {len(node_set)} and the number of "
                  f"nodes pruned is {len(node_list) - len(node_set)}")
            print("**************************")

        return list(node_set)

if __name__ == "__main__":

    abs = ManipAbsBuilder()
    abs.construct_manip_abs(2, ["top", "leftbase", "rightbase", "elsewhere_1"],
                            debug=True, plot=True)
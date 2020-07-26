# a file to construct the manipulation domain
from typing import List, Tuple

from src.graph import FiniteTransSys


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


    # def create_node(self, loc_lst: List[str]):
    #     # a node is of the form (L(cup0),....,L(cup_no_of_objects),L(gripper))
    #     assert len(loc_lst) == self._no_of_objects + 1, " Please enter the locations for all objects" \
    #                                                     " and the gripper location after it"
    #
    #     node = []
    #     for i in loc_lst:
    #         node.append(i)
    #
    #     return tuple(node)


class SysActions:

    def __init__(self, base_actions: List[str]):
        self.__base_actions = base_actions

    def get_base_actions(self):
        return self.__base_actions

    def create_parameterized_action(self, u: tuple, v: tuple):
        raise NotImplementedError


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

    def construct_manip_abs(self, object_count, locations, debug: bool = False):
        manip_graph = ManipAbs("manip_abs", "manip_abs", save_flag=True)
        manip_graph.construct_graph()
        self.construct_nodes(manip_graph, object_count, locations, debug=debug)
        # manip_graph.plot_graph()

    def construct_nodes(self, graph: FiniteTransSys, obj_count: int, locations: List[str], debug: bool = False) -> List[tuple]:
        # construct all the nodes
        node_lst = []

        node_handle = Node()
        node_handle.set_locations_of_interest(locations)
        node_handle.set_no_of_obj(obj_count)
        node_handle.set_gripper_locs()
        node_handle.set_obj_locs()

        # def var_for_loops(locs, num, node_list: List):
        #     if (num > 1):
        #         var_for_loops(locs, num - 1, node_list)
        #     else:
        #         for loc in locs:
        #             n_tuple = node_handle.create_node([loc for i in range(obj_count + 1)])
        #             node_list.append(n_tuple)

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

        # add nodes to the graph
        graph.add_states_from(node_list)

    def sanity_check(self, node_list, node_handle, debug: bool = False) -> List[Tuple]:
        # prune the nodes from the list that are not physically possible
        """
        Physical constraints:
        1. No two objects will have the same locations
        2. No more than one object will be in the gripper's hand |L(cupi) == gripper| = 1
        3. if L(cupi) = gripper => L(G) != "free"
        4. while dropping an object, the location should be free i.e if L(G) == {a location_of_interest - l}
        => for all i L(cupi) != l
        NOTE : I do not add the physical constraint that states that an object should be in base location before
         being at the top. This should captured by the automaton.
        :return: A set of valid nodes that satisfy the above constraints
        """

        def constr_1(node: tuple):
            # enforces constraint # 1., 2., and 4.
            for _l in node_handle._obj_locs:
                if list(node).count(_l) >= 2:
                    return True
            return False

        def constr_2(node: tuple):
            # enforces constraint # 3
            if "gripper" in node:
                if node[-1] == "free":
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

        if debug:
            print("**************************")
            print(f"No. of valid nodes after sanity check is: {len(node_set)} and the number of "
                  f"nodes pruned is {len(node_list) - len(node_set)}")
            print("**************************")

        return list(node_set)

if __name__ == "__main__":

    abs = ManipAbsBuilder()
    abs.construct_manip_abs(3, ["top", "leftbase", "rightbase", "elsewhere_1", "elsewhere_2"], debug=True)
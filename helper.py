import os
import sys
import time
import json
import flask
import warnings
import networkx as nx

from flask import send_file

from copy import deepcopy
from typing import Optional, Union, Dict, Tuple, Generator

from .utls import NpEncoder
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

app = flask.Flask(__name__, static_folder="d3_viz")

@app.route("/")
def static_proxy():
    return app.send_static_file("tree_dfs_all_attrs_scrollable_3.html")

@app.route('/d3_viz/tree_dfs.json')
def serve_json():
    # Assuming your JSON file is in the d3_viz folder
    return send_file('d3_viz/tree_dfs.json')


class InteractiveGraph():
    """
     A Class that contains all the necesary tools to plot a interactive tree using D3.js package
    """
    
    @staticmethod
    def visualize_game(game, source = None, depth_limit: Optional[int] = None, strategy: dict = None, value_dict: dict = None):
        """
         Main method to visualize the gam. We first run a DFS on the game graph and then construct a tree for the given depth limit. 
         Then, we dump the tree data to a json file and serve it over http using D3.js package.
        """
        # call NetworkX and construct Tree for a given depth limit
        if depth_limit is None:
            depth_limit = len(game._graph)

        # run bfs/dfs upto certian depth
        dfs_tree = TreeTraversalMyGame(game=game)
        dfs_tree.tree_traversal(bfs=True, dfs=False, depth_limit=depth_limit, source=source)
        if source is None:
            source = game.get_initial_states()[0][0]
        d = TreeTraversalMyGame.tree_data(G=dfs_tree._tree, root=source, ident="parent", strategy=strategy, value_dict=value_dict)
        
        # write json file
        json.dump(d, open(ROOT_PATH + "/d3_viz/tree_dfs.json", "w"), cls=NpEncoder)
        print("Wrote node-link JSON data to d3_viz repository")
        
        # Serve the file over http to allow for cross origin requests
        print("\nGo to http://localhost:8000 to see the graph\n")
        app.run(port=8000)


class TreeTraversalMyGame():
    
    def __init__(self, game):# strategy: dict = None, value_dict: dict = None):
        self._tree =  nx.DiGraph()
        self.game: TwoPlayerGraph = deepcopy(game)
        # self.strategy: Optional[dict] = strategy
        # self.value_dict: Optional[Dict[str, int]] = value_dict
    
    @staticmethod
    def tree_data(G: nx.DiGraph, root, strategy: dict = None, value_dict: dict = None, ident="id", children="children"):
        """Returns data in tree format that is suitable for JSON serialization
        and use in JavaScript documents.

        Parameters
        ----------
        G : NetworkX graph
        G must be an oriented tree

        root : node
        The root of the tree

        ident : string
            Attribute name for storing NetworkX-internal graph data. `ident` must
            have a different value than `children`. The default is 'id'.

        children : string
            Attribute name for storing NetworkX-internal graph data. `children`
            must have a different value than `ident`. The default is 'children'.

        Returns
        -------
        data : dict
        A dictionary with node-link formatted data.

        Examples
        --------
        >>> from networkx.readwrite import json_graph
        >>> G = nx.DiGraph([(1, 2)])
        >>> data = json_graph.tree_data(G, root=1)

        To serialize with json

        >>> import json
        >>> s = json.dumps(data)

        Notes
        -----
        Node attributes are stored in this format but keys
        for attributes must be strings if you want to serialize with JSON.

        Graph and edge attributes are not stored.

        See Also
        --------
        tree_graph, node_link_data, adjacency_data
        """
        if ident == children:
            raise nx.NetworkXError("The values for `id` and `children` must be different.")
        
        def get_strategy_flag(node, child):
            if isinstance(strategy.get(node), list):
                return True if child in strategy.get(node) else False
            else:
                return True if child == strategy.get(node) else False
        
        
        def add_children(n, G):
            # book keeping
            strategy_flag: bool = False
            
            nbrs = G[n]
            if len(nbrs) == 0:
                return []
            children_ = []
            for child in nbrs:
                if strategy is not None:
                    strategy_flag: bool = get_strategy_flag(n, child)
                
                if value_dict is not None:
                    value = value_dict.get(child)
                    d = {"name": str(child), "edge_name": G[n][child].get('actions'), "label": G.nodes[child].get('ap'), "val": value, "strategy": strategy_flag, "player": G.nodes[child].get('player'), ident: child}
                else:
                    d = {"name": str(child), "edge_name": G[n][child].get('actions'), "label": G.nodes[child].get('ap'), "strategy": strategy_flag, "player": G.nodes[child].get('player'), ident: child}
                c = add_children(child, G)
                if c:
                    d[children] = c
                children_.append(d)
            return children_
        
        if value_dict is not None:
            value = value_dict.get(root)
            return {"name": str(root), "player": G.nodes[root].get('player'), "label": G.nodes[root].get('ap'), "val": value, ident: root, children: add_children(root, G)}
        
        return {"name": str(root), "player": G.nodes[root].get('player'), "label": G.nodes[root].get('ap'), ident: root, children: add_children(root, G)}

    def add_edges(self, ebunch_to_add, **attr) -> None:
            """
            A function to add all the edges in the ebunch_to_add. 
            """
            init_node_added: bool = False
            for e in ebunch_to_add:
                # u and v as 2-Tuple with node and node attributes of source (u) and destination node (v)
                u, v, dd = e
                u_node = u[0]
                u_attr = u[1]
                v_node = v[0]
                v_attr = v[1]
                ddd = {}
                ddd.update(attr)
                ddd.update(dd)

                # add node attributes too
                self._tree.add_node(u_node, **u_attr)
                self._tree.add_node(v_node, **v_attr)

                if u_attr.get('init') and not init_node_added:
                    init_node_added = True
                
                if init_node_added and 'init' in u_attr:
                    del u_attr['init']
                
                # add edge attributes too 
                if not self._tree.has_edge(u_node, v_node):
                    self._tree.add_edge(u_node, v_node, **self.game._graph[u_node][v_node][0])

    def construct_tree_dfs(self, source: None, depth_limit: Union[int, None]) -> Generator[Tuple, Tuple, Dict]:
        """
        This method constructs a tree in a non-recurisve depth first fashion for all plays in the original graph whose depth < depth_limit.
        """
        if source is None:
            source = self.game.get_initial_states()[0][0]
            nodes = [source]

        if depth_limit is None:
            depth_limit = len(self.game)

        visited = set()
        for start in nodes:
            if start in visited:
                continue

            visited.add(start)
            stack = [(start, iter(self.game._graph[start]))] 
            depth_now = 1
            while stack:
                parent, children = stack[-1]
                for child in children:                
                    if child not in visited:
                        yield ((parent), self.game._graph.nodes[parent]), ((child), self.game._graph.nodes[child]), {'weight': 0}
                        visited.add(child)
                        if depth_now < depth_limit:
                            stack.append((child, iter(self.game._graph[child])))
                            depth_now += 1
                            break
                else:
                    stack.pop()
                    depth_now -= 1
    

    def construct_tree_bfs(self, source: None, depth_limit: Union[int, None]) -> Generator[Tuple, Tuple, Dict]:
        """
        This method constructs a tree in a non-recurisve breadth first fashion for all plays in the original graph whose depth < depth_limit.
        """
        if source is None:
            source = self.game.get_initial_states()[0][0]

        if depth_limit is None:
            depth_limit = len(self.game)

        visited = set()
        visited.add(source)
        next_parents_children = [(source, iter(self.game._graph[source]))] 
        depth_now = 0
        while next_parents_children and depth_now < depth_limit:
            this_parents_children = next_parents_children
            next_parents_children = []
            for parent, children in this_parents_children:                
                for child in children:
                    if child not in visited:
                        yield ((parent), self.game._graph.nodes[parent]), ((child), self.game._graph.nodes[child]), {'weight': 0}
                        visited.add(child)
                        next_parents_children.append((child, iter(self.game._graph[child])))
                if len(visited) == len(self.game._graph):
                    return
            depth_now += 1

        print(f"Done with BFS: {len(visited)}") 


    def tree_traversal(self, bfs: bool, dfs: bool = False, depth_limit: Optional[int] = None, source = None) -> nx.DiGraph:
        """
         Parent method to call the traversal.
        """

        start = time.time()
        # self._tree.add_node(self.game.get_initial_states()[0][0])
        if bfs:
            self.add_edges(self.construct_tree_bfs(source=source, depth_limit=depth_limit))
        elif dfs:
            self.add_edges(self.construct_tree_dfs(source=source, depth_limit=depth_limit))
        else:
            warnings.warn("[Error] Please set either bfs or dfs to true")
            sys.exit(-1)
        
        stop = time.time()
        print(f"Time to construct the Tree is: {stop - start:.2f}")
       
'''
Test script to see if wombat code in jupyter nodetobok is wokring or not
List of Dynamics env from minigrid - 

1. Fish and Shipwreck Avoid Agent 
2. Charging station + Agent + Lava
3. Franka Box Packing Extended
4. Fish and Shipwreck
5. Charging Station 
6. Corridorr to charging station
7. Dynamic charging station
'''

import os
import sys
import gym
import copy
import time
import warnings
import numpy as np
from pathlib import Path

EXAMPLE_DIR = os.getcwd()
os.chdir(os.path.join('..'))
PROJECT_DIR = os.getcwd()
print('EXAMPLE_DIR: ', EXAMPLE_DIR)
print('PROJECT_DIR: ', PROJECT_DIR)

sys.path.append(os.path.join(PROJECT_DIR, 'src'))
from src.graph import Graph
from src.graph import graph_factory
from src.config import ROOT_PATH
from src.strategy_synthesis.multiobjective_solver import MultiObjectiveSolver
from src.simulation.simulator import Simulator

# from src.g

sys.path.append(os.path.join(PROJECT_DIR, 'wombats'))
from wombats.systems.minigrid import GYM_MONITOR_LOG_DIR_NAME
from wombats.systems.minigrid import DynamicMinigrid2PGameWrapper, MultiAgentMiniGridEnv

from src.strategy_synthesis.value_iteration import ValueIteration
from src.strategy_synthesis.regret_str_synthesis import RegretMinimizationStrategySynthesis as RegMinStrSyn
from src.strategy_synthesis.best_effort_syn import QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn
from src.strategy_synthesis.best_effort_safe_reach import QualitativeSafeReachBestEffort, QuantitativeSafeReachBestEffort


DIR = EXAMPLE_DIR
Graph.automata_data_dir = DIR

debug = True

# env_id = 'MiniGrid-ToyCorridorLava-v0'
env_id = "MiniGrid-CorridorLava-v0"
pdfa_config_yaml="config/PDFA_charging_station"
player_steps = {'sys': [1], 'env': [1]}
# player_steps = {'sys': [1, 2], 'env': [1]}
player_steps = {'sys': [1], 'env': [1]}

load_game_from_file = False
plot_minigrid = False
plot_pdfa = True
plot_product = False
finite = True
view = False
save_flag = True
format = 'png'

stochastic = False
adversarial = True
plot_strategies=False
plot_graph_with_strategy = False
plot_graph_with_pareto = False
plot_pareto = True
speedup = True

env_filename = os.path.join(DIR, 'plots', 'gym_env.png')
Path(os.path.split(env_filename)[0]).mkdir(parents=True, exist_ok=True)
env_dpi = 300



def build_LTL_automaton(formula: str, debug: bool = False, plot: bool = False, use_alias: bool = False):
    """
    A method to construct automata using the regret_synthesis_tool.
    """
    if not isinstance(formula, str):
        warnings.warn("Please make sure the input formula is of type string.")

    _ltl_automaton = graph_factory.get('DFA',
                                        graph_name="minigrid_ltl",
                                        config_yaml="/config/minigrid_ltl",
                                        save_flag=True,
                                        sc_ltl=formula,
                                        use_alias=use_alias,
                                        plot=plot)

    if debug:
        print(f"The pddl formula is : {formula}")

    return _ltl_automaton


def build_product(dfa, trans_sys, plot: bool = False):
    _product_automaton = graph_factory.get("ProductGraph",
                                           graph_name="minigrid_product_graph",
                                           config_yaml="/config/minigrid_product_graph",
                                           trans_sys=trans_sys,
                                           automaton=dfa,
                                           save_flag=True,
                                           prune=False,
                                           debug=False,
                                           absorbing=True,
                                           finite=False,
                                           plot=plot,
                                           pdfa_compose=False)

    print("Done building the Product Automaton")

    # Add the accepting state "accept_all" in the product graph with player = "eve"
    # should technically be only one if absorbing is true
    _states = _product_automaton.get_accepting_states()

    return _product_automaton


def initialize_edge_labels_on_fancy_graph(two_player_graph):

    edges = set(two_player_graph._graph.edges())

    for edge in edges:
        for i, edge_data in two_player_graph._graph[edge[0]][edge[1]].items():
            actions = edge_data.get('symbols')
            weights = edge_data.get('weight')
            # label = copy.deepcopy(weights)
            # label.update({'actions': actions})
            two_player_graph._graph[edge[0]][edge[1]][i]['actions'] = actions[0]
            two_player_graph._graph[edge[0]][edge[1]][i]['weight'] = weights[0]


def build_minigrid_game(env_snap: bool = False):
    # OpenAI Minigrid Env
    env = gym.make(env_id)
    env = DynamicMinigrid2PGameWrapper(
            env,
            player_steps=player_steps,
            monitor_log_location=os.path.join(DIR, GYM_MONITOR_LOG_DIR_NAME))
    env.reset()
    if env_snap:
        env.render_notebook(env_filename, env_dpi)

    file_name = env_id + 'Game'
    filepath = os.path.join(DIR, 'config', file_name)
    config_yaml = os.path.relpath(filepath, ROOT_PATH)

    # Game Construction
    start = time.time()
    two_player_graph = graph_factory.get('TwoPlayerGraph',
                                graph_name='TwoPlayerGame',
                                config_yaml=config_yaml,
                                from_file=load_game_from_file,
                                minigrid=env,
                                save_flag=save_flag,
                                plot=plot_minigrid,
                                view=view,
                                format=format)
    end = time.time()
    print(f"Two Player Graph Construction (s): {end - start}")

    # add labels to the graph 
    initialize_edge_labels_on_fancy_graph(two_player_graph)


    return two_player_graph, env


def simulate_strategy(env, game, sys_actions, iterations: int = 100):
    """
    A function to simulate the synthesize strategy. Available env_action 'random' and 'interactive'.
    """
    sim = Simulator(env, game)
    sim.run(iterations=iterations,
            sys_actions=sys_actions,
            env_strategy='random',
            render=False,
            record_video=False)
    sim.get_stats()


def compute_strategy(strategy_type: str, game, debug: bool = False, plot: bool = False, reg_factor: float = 1.25):
    """
     A method that call the appropriate strategy synthesis class nased on the user input. 

     Valid strategy_type: Min-Max, Min-Min, Regret, BestEffortQual, BestEffortQuant, BestEffortSafeReachQual, BestEffortSafeReachQuant
    """
    valid_str_syn_algos = ["Min-Max", "Min-Min", "Regret", "BestEffortQual", "BestEffortQuant", "BestEffortSafeReachQual", "BestEffortSafeReachQuant"]

    if strategy_type == "Min-Max":
        strategy_handle = ValueIteration(game, competitive=True)
        strategy_handle.solve(debug=debug, plot=plot)

    elif strategy_type == "Min-Min":
        strategy_handle = ValueIteration(game, competitive=False)
        strategy_handle.solve(debug=debug, plot=plot)
    
    elif strategy_type == "Regret":
        strategy_handle = RegMinStrSyn(game)
        strategy_handle.edge_weighted_arena_finite_reg_solver(reg_factor=reg_factor,
                                                              purge_states=True,
                                                              plot=plot)
    
    elif strategy_type == "BestEffortQual":
        strategy_handle = QualitativeBestEffortReachSyn(game=game, debug=debug)    
        strategy_handle.compute_best_effort_strategies(plot=plot)
    
    # My propsoed algorithms
    elif strategy_type == "BestEffortQuant":
        strategy_handle = QuantitativeBestEffortReachSyn(game=game, debug=debug)
        strategy_handle.compute_best_effort_strategies(plot=plot)
    
    elif strategy_type == "BestEffortSafeReachQual":
        strategy_handle = QualitativeSafeReachBestEffort(game=game, debug=debug)
        strategy_handle.compute_best_effort_strategies(plot=plot)
    
    elif strategy_type == "BestEffortSafeReachQuant":
        strategy_handle = QuantitativeSafeReachBestEffort(game=game, debug=debug)
        strategy_handle.compute_best_effort_strategies(plot=plot)

    else:
        warnings.warn(f"[Error] Please enter a valid Strategy Synthesis variant:[ {', '.join(valid_str_syn_algos)} ]")
        sys.exit(-1)

    return strategy_handle



if __name__ == '__main__':

    # build two player game
    game, env = build_minigrid_game()

    # get all the aps, and the player
    set_ap = set({})
    set_players = set({})
    init_states = set({})
    
    # set of valid attributes - ap, player, init
    for s in game._graph.nodes():
        set_ap.add(game.get_state_w_attribute(s, 'ap'))
        set_players.add(game.get_state_w_attribute(s, 'player'))
    

    # get edge weight sets
    _edge_weights = set({})
    action_set = set({})
    for _s in game._graph.nodes():
        for _e in game._graph.out_edges(_s):
            if game._graph[_e[0]][_e[1]][0].get('weight', None):
                _edge_weights.add(game._graph[_e[0]][_e[1]][0]["weight"])
    
    # set edge weight sets
    for _s in game._graph.nodes():
        for _e in game._graph.out_edges(_s):
            game._graph[_e[0]][_e[1]][0]["weight"] = 1 if game._graph.nodes(data='player')[_s] == 'eve' else 0

    print(f"Set of APs: {set_ap}")
    print(f"Set of Players: {set_players}")

    # build DFA
    formula = 'F(floor_green_open)'
    dfa = build_LTL_automaton(formula=formula,
                              debug= False,
                              plot=False,
                              use_alias=False)


    # build the product
    dfa_game = build_product(dfa=dfa,
                             trans_sys=game,
                             plot=False)


    # synthesize strategy - Min-Max, Min-Min
    # valid_str_syn_algos = ["Min-Max", "Min-Min", "Regret", "BestEffortQual", "BestEffortQuant", "BestEffortSafeReachQual", "BestEffortSafeReachQuant"]
    start = time.time()
    valid_str_syn_algos = ["Min-Max"]
    for st in valid_str_syn_algos:
        strategy_handle = compute_strategy(strategy_type=st, game=dfa_game, debug=False, plot=False)
    end = time.time()
    print(f"Done Synthesizing a strategy: {end-start:0.2f} seconds")
    
    # print(strategy_handle)
    simulate_str: bool = True
    if simulate_str:
        player = 'sys'
        SYS_ACTIONS = []
        for multiactions in env.player_actions[player]:
            action_strings = []
            for agent, actions in zip(env.unwrapped.agents, multiactions):
                action_string = []
                for action in actions:
                    if action is None or np.isnan(action):
                        continue
                    a_str = agent.ACTION_ENUM_TO_STR[action]
                    action_string.append(a_str)
                action_strings.append(tuple(action_string))
            action_strs = action_strings[0] if player == 'sys' else action_strings[1:]
            SYS_ACTIONS.append(tuple(action_strs))
        
        print(SYS_ACTIONS)

        # hard code actions to go to green region
        two_right_steps = "east_east__None"
        one_right_step = "east__None"
        up_step = "north__None"
        down_step = "south__None"

        # for two step envs
        # up_step = ('north',)
        # down_step = ('south',)
        # one_right_step = ('east',)
        # one_left_step = ('west',)
        # STRATEGY = [two_right_steps, one_right_step, up_step, two_right_steps, down_step, two_right_steps, one_right_step]

        # for one step envs
        print("Starting to Roll out strategy")
        start = time.time()
        STRATEGY = [one_right_step, one_right_step, one_right_step, up_step, one_right_step, one_right_step, down_step, one_right_step, one_right_step, one_right_step]
        simulate_strategy(env=env, game=dfa_game, sys_actions=STRATEGY)
        print("Done Rolling out strategy")
        end = time.time()
        print(f"Done Rolling out strategy: {end-start:0.2f} seconds")
    


import os
import sys
import gym
import abc
import random
import warnings
import argparse
import numpy as np

from typing import Tuple, Optional, Dict, Union, Any

# import wombats packages
from wombats.systems import StaticMinigridTSWrapper
from wombats.systems.minigrid import DynamicMinigrid2PGameWrapper, MultiAgentMiniGridEnv
from wombats.systems.minigrid import GYM_MONITOR_LOG_DIR_NAME
from wombats.automaton import active_automata
from wombats.automaton import MinigridTransitionSystem

# import local packages
from src.graph import Graph
from src.graph import graph_factory
from src.graph import MiniGrid
from src.graph import FiniteTransSys
from src.graph import DFAGraph
from src.graph import ProductAutomaton
from src.graph import TwoPlayerGraph

# import available str synthesis methods
from src.strategy_synthesis import ReachabilitySolver
from src.strategy_synthesis import MultiObjectiveSolver
from src.prism import PrismInterfaceForTwoPlayerGame


Strategy = Any # TODO: import

# directory where we will be storing all the configuration files related to graphs
# os.chdir(os.path.join('..', '..'))
DIR = os.path.dirname(os.path.abspath(__file__))
Graph.automata_data_dir = DIR


def execute_str(wombats_minigrid_TS, controls):
    wombats_minigrid_TS.run(controls, record_video=True, show_steps=True)


def build_game_from_minigrid(
    env_id: str,
    pdfa_config_yaml: str,
    sc_ltl: str="F (floor_green_open)",
    plot_minigrid: bool = False,
    plot_pdfa: bool = False,
    plot_product: bool = False,
    finite: bool = True,
    view: bool = False,
    format: str = 'svg'):

    env = gym.make(env_id)

    env = DynamicMinigrid2PGameWrapper(
        env,
        actions_types=['simple_static', 'simple_static'],
        monitor_log_location=os.path.join(DIR, GYM_MONITOR_LOG_DIR_NAME))
    env.reset()
    # env.render()
    # env.close()

    file_name = env_id + '_Game'
    game = graph_factory.get('TwoPlayerGraph',
                             graph_name='TwoPlayerGame',
                             config_yaml=f'/config/{file_name}',
                             minigrid=env,
                             save_flag=True,
                             plot=plot_minigrid,
                             view=view,
                             format=format)

    pdfa = graph_factory.get(
        'PDFA',
        graph_name="pdfa",
        config_yaml=pdfa_config_yaml,
        save_flag=True,
        plot=plot_pdfa)

    product_automaton = graph_factory.get('ProductGraph',
        graph_name='ProductAutomaton',
        config_yaml='/config/product_automaton',
        trans_sys=game,
        automaton=pdfa,
        save_flag=True,
        prune=False,
        debug=False,
        absorbing=True,
        finite=finite,
        plot=plot_product,
        integrate_accepting=True,
        view=view,
        format=format)

    return product_automaton, env


def build_game_from_product(
    ts_config_yaml: str,
    pdfa_config_yaml: str,
    finite: bool = True,
    plot_ts: bool = False,
    plot_pdfa: bool = False,
    plot_product: bool = False,
    prune: bool = False,
    plot_auto_graph: bool = False,
    plot_trans_graph: bool = False,
    integrate_accepting: bool = True,
    use_trans_sys_weights: bool = True,
    view: bool = False,
    format: str = 'svg'):

    trans_sys = graph_factory.get('TS',
                                  raw_trans_sys=None,
                                  graph_name="trans_sys",
                                  config_yaml=ts_config_yaml,
                                  from_file=True,
                                  pre_built=False,
                                  save_flag=True,
                                  plot=plot_ts,
                                  human_intervention=0,
                                  finite=finite)

    pdfa = graph_factory.get('PDFA',
                             graph_name="pdfa",
                             config_yaml=pdfa_config_yaml,
                             save_flag=True,
                             plot=plot_pdfa)

    product_automaton = graph_factory.get('ProductGraph',
                                          graph_name='ProductAutomaton',
                                          config_yaml='/config/product_automaton',
                                          trans_sys=trans_sys,
                                          automaton=pdfa,
                                          save_flag=True,
                                          prune=prune,
                                          debug=True,
                                          absorbing=True,
                                          finite=finite,
                                          plot=plot_product,
                                          plot_auto_graph=plot_auto_graph,
                                          plot_trans_graph=plot_trans_graph,
                                          integrate_accepting=integrate_accepting,
                                          use_trans_sys_weights = use_trans_sys_weights,
                                          view=view,
                                          format=format)

    return product_automaton


def multi_objective_two_player_game(
    game: TwoPlayerGraph,
    env: MultiAgentMiniGridEnv = None,
    stochastic: bool = False,
    adversarial: bool = True,
    plot_strategies: bool = False,
    plot_graph_with_strategy: bool = False,
    plot_pareto: bool = True,
    debug: bool = False,
    use_prism: bool = False,
    view: bool = False,
    format: str = 'svg'):

    if use_prism:

        prism_interface = PrismInterfaceForTwoPlayerGame(use_docker=True)

        prism_interface.run_prism(
            game,
            pareto=True,
            paretoepsilon=0.00001,
            plot=plot_pareto,
            explicit=True,
            debug=True)

        for pareto_point in prism_interface.pareto_points:
            prism_interface.run_prism(
                game, pareto_point=pareto_point, debug=debug)

    else:

        solver = MultiObjectiveSolver(game,
                                    epsilon=1e-7,
                                    max_iteration=300,
                                    stochastic=stochastic,
                                    adversarial=adversarial)
        # solver.solve(stochastic=True, adversarial=True, plot_strategies=True, bound=[5.6, 5.6])
        solver.solve(plot_strategies=plot_strategies,
                     plot_graph_with_strategy=plot_graph_with_strategy,
                     debug=debug,
                     view=view,
                     format=format)

        if env is not None:
            for pp in solver.get_pareto_points():
                strategy = solver.get_a_strategy_for(pp)
                run_turn_based_game(env, strategy, debug)


def run_turn_based_game(env: MultiAgentMiniGridEnv, strategy: Strategy, debug: bool = False,
                        render: bool = True):

    accepted = False
    curr_agent = 'sys'
    n_agent = env.n_agent

    env._toggle_video_recording(record_video=True)
    env.reset()
    # In case of stochastic strategy, it is a memory based strategy.
    # Therefore, it needs to be reset.
    strategy.reset()
    if render:
        env.render()
    action_str = None

    action_strs = [s.split('.')[1] for s in list(map(str, env.agents[0].actions))]
    prev_state =  strategy.curr_state[0]

    while not accepted:

        multi_agent_actions = []

        if curr_agent == 'sys':
            # TODO: Strategy takes current state and action
            # TODO: and outputs "actions", if done, it's done
            a_strs, accepted = strategy.step(action_str)    # Ideally, just states

            if accepted:
                break

            for a_str in a_strs:
                index = action_strs.index(a_str)
                action = list(env.agents[0].actions)[index]
                multi_agent_action = [None] * n_agent
                multi_agent_action[0] = action
                multi_agent_actions.append(multi_agent_action)
        else:
            # Env takes a random action
            # TODO: This should be inside the env.
            # TODO: It should have both rand, adversarial strategy
            # TODO: We might need to compute adv strat from pareto points
            index = random.randint(1, n_agent-1)
            env_agent = env.agents[index]
            action = random.choice(list(env_agent.actions))
            action_str = (action_strs[action], ) # TODO: multiple actions to a tuple
            a_strs = [action_str]
            multi_agent_action = [None] * n_agent
            multi_agent_action[index] = action
            multi_agent_actions.append(multi_agent_action)

        # TODO: env should take in multiple actions
        # TODO: Inside, it can choose to take actions in turn or concurrently
        for multi_agent_action in multi_agent_actions:
            # TODO: Env should output env actions, rewards, next state, done, info
            env.step(multi_agent_action)

            if render:
                env.render()

        curr_agent = 'env' if curr_agent == 'sys' else 'sys'
        state = env._state_to_node_name(env.get_state(), curr_agent)
        if debug:
            print(f'{prev_state}, [{a_strs}] -> Current State: {state}')
        prev_state = state

    env.close()
    env._toggle_video_recording()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finite", action="store_true")
    parser.add_argument("--ts", type=str, default='product',
        choices=['gym_minigrid', 'product'])
    parser.add_argument("--example", type=str, default='SysLoop',
        choices=['SysLoop', 'EnvLoop', 'StrategyDiff', 'ElevEscStairs', 'SimpleLoop'])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cooperative", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--plot_ts", action="store_true")
    parser.add_argument("--plot_pdfa", action="store_true")
    parser.add_argument("--plot_product", action="store_true")
    parser.add_argument("--plot_minigrid", action="store_true")
    parser.add_argument("--plot_strategies", action="store_true")
    parser.add_argument("--plot_graph_with_strategy", action="store_true")
    parser.add_argument("--plot_pareto", action="store_true")
    parser.add_argument("--plot_auto_graph", action="store_true")
    parser.add_argument("--plot_trans_graph", action="store_true")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--format", type=str, default='svg',
        choices=['svg', 'png', 'jpeg', 'pdf'])

    return parser.parse_args()


def choose_minigrid_config():
    # sc_ltl="!(lava_red_open) U(carpet_yellow_open) &(!(lava_red_open) U (water_blue_open))",
    # sc_ltl="!(lava_red_open) U (water_blue_open)",
    # sc_ltl="!(lava_red_open) U (goal_green_open)",
    #   sc_ltl="F (goal_green_open)",
    #   sc_ltl="F (floor_purple_open) & F (floor_green_open) & (!(lava_red_open) U (floor_green_open)) & (!(lava_red_open) U (floor_purple_open))",

    # env_id = 'MiniGrid-FourGrids-v0'
    # env_id = 'MiniGrid-ChasingAgent-v0'
    # env_id = 'MiniGrid-ChasingAgentInSquare4by4-v0'
    env_id = 'MiniGrid-ChasingAgentInSquare3by3-v0'
    sc_ltl="F (floor_green_open) & F (agent_blue_right)"
    pdfa_config_yaml="/config/PDFA_Fish_and_Shipwreck"

    return env_id, sc_ltl, pdfa_config_yaml


def choose_product_config(example: str = 'SysLoop'):

    if example == 'SysLoop':

        use_trans_sys_weights = True
        pdfa_config = "/config/PDFA_onegoal"
        ts_config = "/config/Game_sys_loops"

    elif example == 'EnvLoop':

        use_trans_sys_weights = True
        pdfa_config = "/config/PDFA_onegoal"
        ts_config = "/config/Game_env_loops"

    elif example == 'StrategyDiff':

        use_trans_sys_weights = True
        pdfa_config = "/config/PDFA_onegoal"
        ts_config = "/config/Game_all_problems"
        # self._ts_config = "/config/Game_one_in_three_pareto_points"
        # self._ts_config = "/config/Game_three_pareto_points"

    elif example == 'ElevEscStairs':

        use_trans_sys_weights = False
        pdfa_config = "/config/PDFA_threegoals"
        ts_config = "/config/Game_elev_esc_stairs"

    elif example == 'SimpleLoop':

        use_trans_sys_weights = False
        pdfa_config = "/config/PDFA_twogoals"
        ts_config = "/config/Game_simple_loop"

    # self._ts_config = "/config/Game_two_goals"
    # self._ts_config = "/config/Game_simple_loop"
    # self._ts_config = "/config/Game_two_goals_self_loop"
    # self._ts_config = "/config/Game_all_problems"

    # if multiple_accepting:
    #     self._auto_config = "/config/PDFA_multiple_accepting"
    # else:
    #     # self._auto_config = "/config/PDFA_twogoals"
    #     self._auto_config = "/config/PDFA_onegoal"

    return ts_config, pdfa_config, use_trans_sys_weights


if __name__ == "__main__":

    args = parse_arguments()

    env_id, sc_ltl, pdfa_config_yaml = choose_minigrid_config()

    ts_config, pdfa_config, use_trans_sys_weights = choose_product_config(args.example)

    if args.ts == 'gym_minigrid':
        game, env = build_game_from_minigrid(
                env_id=env_id,
                pdfa_config_yaml=pdfa_config_yaml,
                sc_ltl=sc_ltl,
                plot_minigrid=args.plot_minigrid,
                plot_pdfa=args.plot_pdfa,
                plot_product=args.plot_product,
                finite=args.finite,
                view=args.view,
                format=args.format)
    elif args.ts == 'product':
        game = build_game_from_product(
            ts_config,
            pdfa_config,
            use_trans_sys_weights=use_trans_sys_weights,
            plot_ts=args.plot_ts,
            plot_pdfa=args.plot_pdfa,
            plot_product=args.plot_product,
            plot_auto_graph=args.plot_auto_graph,
            plot_trans_graph=args.plot_trans_graph,
            finite=args.finite,
            view=args.view,
            format=args.format)
        env = None
    else:
        warnings.warn("Please ensure at-least one of the flags is True")
        sys.exit(-1)

    multi_objective_two_player_game(
        game=game,
        env=env,
        stochastic=args.stochastic,
        adversarial=not args.cooperative,
        plot_strategies=args.plot_strategies,
        plot_graph_with_strategy=args.plot_graph_with_strategy,
        plot_pareto=args.plot_pareto,
        debug=args.debug,
        view=args.view,
        format=args.format
    )

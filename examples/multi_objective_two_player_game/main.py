import os
import sys
import gym
import warnings
import argparse
import time
from typing import  Dict, Any

# import wombats packages
from wombats.systems.minigrid import DynamicMinigrid2PGameWrapper, MultiAgentMiniGridEnv
from wombats.systems.minigrid import GYM_MONITOR_LOG_DIR_NAME

# import local packages
from src.graph import Graph
from src.graph import graph_factory
from src.graph import TwoPlayerGraph
from src.config import ROOT_PATH

# import available str synthesis methods
from src.strategy_synthesis import MultiObjectiveSolver
from src.prism import PrismInterfaceForTwoPlayerGame
from src.simulation.simulator import Simulator


Strategy = Any # TODO: import

# directory where we will be storing all the configuration files related to graphs
# os.chdir(os.path.join('..', '..'))
DIR = os.path.dirname(os.path.abspath(__file__))
print('Now at directory: ', DIR)
Graph.graph_dir = DIR


def build_game_from_minigrid(
    env_id: str,
    pdfa_config_yaml: str,
    player_steps: Dict ,
    # from_file: bool = True,
    from_file: bool = False,
    plot_minigrid: bool = False,
    plot_pdfa: bool = False,
    plot_product: bool = False,
    finite: bool = True,
    view: bool = False,
    format: str = 'svg'):

    # PDFA
    pdfa = graph_factory.get(
        'PDFA',
        graph_name="pdfa",
        config_yaml=pdfa_config_yaml,
        save_flag=True,
        plot=plot_pdfa)

    # OpenAI Minigrid Env
    env = gym.make(env_id)
    env = DynamicMinigrid2PGameWrapper(
        env,
        player_steps=player_steps,
        monitor_log_location=os.path.join(DIR, GYM_MONITOR_LOG_DIR_NAME))
    env.reset()

    file_name = env_id + 'Game'
    filepath = os.path.join(DIR, 'config', file_name)
    config_yaml = os.path.relpath(filepath, ROOT_PATH)

    # Game Construction
    start = time.time()
    game = graph_factory.get('TwoPlayerGraph',
                             graph_name='TwoPlayerGame',
                             config_yaml=config_yaml,
                             from_file=from_file,
                             minigrid=env,
                             save_flag=True,
                             plot=plot_minigrid,
                             view=view,
                             format=format)
    end = time.time()
    print(f'Game Extraction took {end-start:.2f} seconds')

    # Product Game Construction
    file_name = env_id + 'ProductAutomaton'
    config_yaml = os.path.join(DIR, 'config', file_name)
    config_yaml = None
    start = time.time()

    product_automaton = graph_factory.get('ProductGraph',
        graph_name='ProductAutomaton',
        config_yaml=config_yaml,
        trans_sys=game,
        automaton=pdfa,
        save_flag=True,
        prune=False,
        debug=False,
        absorbing=True,
        finite=finite,
        from_file=from_file,
        plot=plot_product,
        integrate_accepting=True,
        view=view,
        format=format)
    end = time.time()
    print(f'Product Construction took {end-start:.2f} seconds')

    return product_automaton, env


def build_game_from_product(
    ts_config_yaml: str,
    pdfa_config_yaml: str,
    finite: bool = True,
    plot_ts: bool = False,
    plot_pdfa: bool = False,
    plot_product: bool = False,
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
                                          config_yaml='config/product_automaton',
                                          from_file=False,
                                          trans_sys=trans_sys,
                                          automaton=pdfa,
                                          save_flag=True,
                                          prune=False,
                                          debug=True,
                                          absorbing=True,
                                          finite=finite,
                                          plot=plot_product,
                                          integrate_accepting=True,
                                          use_trans_sys_weights = use_trans_sys_weights,
                                          view=view,
                                          format=format)

    return product_automaton


def multi_objective_two_player_game(
    game: TwoPlayerGraph,
    stochastic: bool = False,
    adversarial: bool = True,
    plot_strategies: bool = False,
    plot_graph_with_strategy: bool = False,
    plot_graph_with_pareto: bool = False,
    plot_pareto: bool = True,
    speedup_pareto_computation: bool = True,
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

        # Evaluate for each pareto point for exporting images
        for pareto_point in prism_interface.pareto_points:
            prism_interface.run_prism(
                game, pareto_point=pareto_point, debug=debug)

        solver = None

    else:

        solver = MultiObjectiveSolver(game,
                                    epsilon=1e-7,
                                    max_iteration=300,
                                    stochastic=stochastic,
                                    adversarial=adversarial)
        solver.solve(plot_strategies=plot_strategies,
                     plot_graph_with_strategy=plot_graph_with_strategy,
                     plot_graph_with_pareto=plot_graph_with_pareto,
                     plot_pareto=plot_pareto,
                     speedup=speedup_pareto_computation,
                     debug=debug,
                     view=view,
                     format=format)

        return solver


def run_evaluation(solver: MultiObjectiveSolver,
    env: MultiAgentMiniGridEnv = None,
    iterations: int = 10):

    if solver is None or env is None:
        return

    for pp in solver.get_pareto_points():
        strategy = solver.get_a_strategy_for(pp)

        print('-'*100)
        print(f"Evaluate for a pareto point {pp}")
        print('-'*100)

        sim = Simulator(env, game)
        sim.run(iterations=iterations,
                sys_strategy=strategy,
                render=iterations==1,
                record_video=iterations<=15)
        print(sim.get_stats())


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--finite", action="store_true")
    parser.add_argument("--ts", type=str, default='product',
        choices=['gym_minigrid', 'product'])
    parser.add_argument("--example", type=str, default='SysLoop',
        choices=['SysLoop', 'EnvLoop', 'StrategyDiff', 'ElevEscStairs', 'SimpleLoop', 'FishShipwreck', 'ChargingStation'])
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
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--speedup_pareto_computation", action="store_true")
    parser.add_argument("--format", type=str, default='svg',
        choices=['svg', 'png', 'jpeg', 'pdf'])

    return parser.parse_args()


def choose_minigrid_config(example):
    if example == 'FishShipwreck':

        env_id = 'MiniGrid-FourGrids-v0'
        # env_id = 'MiniGrid-ChasingAgent-v0'
        # env_id = 'MiniGrid-ChasingAgentInSquare4by4-v0'
        # env_id = 'MiniGrid-ChasingAgentInSquare3by3-v0'
        # env_id = 'MiniGrid-ChasingAgentIn4Square-v0'
        pdfa_config_yaml="config/PDFA_Fish_and_Shipwreck"
        player_steps = {'sys': [1, 2], 'env': [1]}

    elif example == 'ChargingStation':

        env_id = 'MiniGrid-FloodingLava-v0'
        pdfa_config_yaml="config/PDFA_charging_station"
        # player_steps = {'sys': [1], 'env': [1]}
        player_steps = {'sys': [1], 'env': [1]}

    elif example == 'DynamicChargingStation':

        env_id = 'MiniGrid-DynamicLava-v0'
        pdfa_config_yaml="config/PDFA_charging_station"
        # player_steps = {'sys': [1], 'env': [1]}
        player_steps = {'sys': [2], 'env': [2]}


    return env_id, pdfa_config_yaml, player_steps


def choose_product_config(example: str = 'SysLoop'):

    if example == 'SysLoop':

        use_trans_sys_weights = True
        pdfa_config = "config/PDFA_onegoal"
        ts_config = "config/Game_sys_loops"
        ts_config = "config/Game_sys_loops_longerdistance"

    elif example == 'EnvLoop':

        use_trans_sys_weights = True
        pdfa_config = "config/PDFA_onegoal"
        ts_config = "config/Game_env_loops"

    elif example == 'StrategyDiff':

        use_trans_sys_weights = True
        pdfa_config = "config/PDFA_onegoal"
        # ts_config = "config/Game_all_problems"
        use_trans_sys_weights = False
        ts_config = "config/Game_one"
        # ts_config = "config/Game_three_pareto_points"

    elif example == 'ElevEscStairs':

        use_trans_sys_weights = False
        pdfa_config = "config/PDFA_threegoals"
        ts_config = "config/Game_elev_esc_stairs"

    elif example == 'SimpleLoop':

        use_trans_sys_weights = False
        pdfa_config = "config/PDFA_twogoals"
        ts_config = "config/Game_simple_loop"

    # self._ts_config = "config/Game_two_goals"
    # self._ts_config = "config/Game_simple_loop"
    # self._ts_config = "config/Game_two_goals_self_loop"
    # self._ts_config = "config/Game_all_problems"

    # if multiple_accepting:
    #     self._auto_config = "/config/PDFA_multiple_accepting"
    # else:
    #     # self._auto_config = "/config/PDFA_twogoals"
    #     self._auto_config = "/config/PDFA_onegoal"

    return ts_config, pdfa_config, use_trans_sys_weights


if __name__ == "__main__":

    args = parse_arguments()

    if args.evaluation:

        # Load a solver OR a game (w/ pareto points computed) and an env
            # Think about the behavior. How to decide which one to choose
        # run a simulation

        # plot 2D pareto points, strategies, game_w_strategies
        pass

    else:

        if args.ts == 'gym_minigrid':

            env_id, pdfa_config_yaml, player_steps = choose_minigrid_config(args.example)
            game, env = build_game_from_minigrid(
                    env_id=env_id,
                    pdfa_config_yaml=pdfa_config_yaml,
                    player_steps=player_steps,
                    plot_minigrid=args.plot_minigrid,
                    plot_pdfa=args.plot_pdfa,
                    plot_product=args.plot_product,
                    finite=args.finite,
                    view=args.view,
                    format=args.format)

        elif args.ts == 'product':

            ts_config, pdfa_config, use_trans_sys_weights = choose_product_config(args.example)
            game = build_game_from_product(
                ts_config,
                pdfa_config,
                use_trans_sys_weights=use_trans_sys_weights,
                plot_ts=args.plot_ts,
                plot_pdfa=args.plot_pdfa,
                plot_product=args.plot_product,
                finite=args.finite,
                view=args.view,
                format=args.format)
            env = None

        else:
            warnings.warn("Please ensure at-least one of the flags is True")
            sys.exit(-1)

        solver = multi_objective_two_player_game(
            game,
            stochastic=args.stochastic,
            adversarial=not args.cooperative,
            plot_strategies=args.plot_strategies,
            plot_graph_with_strategy=args.plot_graph_with_strategy,
            plot_pareto=args.plot_pareto,
            debug=args.debug,
            view=args.view,
            format=args.format
        )

        run_evaluation(
            solver,
            env,
            args.iterations)

import sys
import warnings

from typing import Union

# import local packages
from src.graph import graph_factory
from src.graph import FiniteTransSys
from src.graph import TwoPlayerGraph

# import available game models
from graph_examples.simple_two_player_games import *
from graph_examples.adm_two_player_games import *


# import available str synthesis methods
from src.strategy_synthesis.regret_str_synthesis \
    import RegretMinimizationStrategySynthesis as RegMinStrSyn
from src.strategy_synthesis.adversarial_game import ReachabilityGame as ReachabilitySolver
from src.strategy_synthesis.safety_game import SafetyGame
from src.strategy_synthesis.cooperative_game import CooperativeGame
from src.strategy_synthesis.iros_solver import IrosStrategySynthesis as IrosStrSolver
from src.strategy_synthesis.value_iteration import ValueIteration, PermissiveValueIteration
from src.strategy_synthesis.best_effort_syn import QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn, \
      QuantitativeHopefullAdmissibleReachSyn
from src.strategy_synthesis.adm_str_syn import QuantitativeNaiveAdmissible, QuantitativeGoUAdmissible, QuantitativeGoUAdmissibleWinning

def compute_bounded_winning_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph],
                                energy_bound: int = 0,
                                debug: bool = False,
                                print_str: bool = False):

    iros_solver = IrosStrSolver(game=trans_sys, energy_bound=energy_bound, plot_modified_game=False)
    _start_state = trans_sys.get_initial_states()[0][0]
    if iros_solver.solve(debug=debug):
        print(f"There EXISTS a winning strategy from the  initial game state {_start_state} "
              f"with max cost of {iros_solver.str_map[_start_state]['cost']}")

    else:
        print(f"There DOES NOT exists a winning strategy from the  initial game state {_start_state} "
              f"with max cost of {iros_solver.str_map[_start_state]['cost']}")

    if print_str:
        iros_solver.print_map_dict()


def compute_winning_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph],
                        debug: bool = False,
                        permissive_strategies: bool = False,
                        print_winning_regions: bool = False,
                        print_str: bool = False,
                        plot: bool = False):

    reachability_game_handle = ReachabilitySolver(game=trans_sys, debug=debug)
    reachability_game_handle.reachability_solver()

    if print_winning_regions:
        reachability_game_handle.print_winning_region()

    if print_str:
        reachability_game_handle.print_winning_strategies()

    if reachability_game_handle.is_winning():
        print("Assuming Env to be adversarial, sys CAN force a visit to the accepting states")
    else:
        print("Assuming Env to be adversarial, sys CANNOT force a visit to the accepting states")
    
    if plot:
        reachability_game_handle.plot_graph(with_strategy=True)


def play_min_max_game(trans_sys: Union[FiniteTransSys, TwoPlayerGraph],
                      debug: bool = False,
                      plot: bool = False,
                      competitive: bool = True,
                      permissive_strategies: bool = False):
    
    if permissive_strategies:
        vi_handle = PermissiveValueIteration(game=trans_sys, competitive=competitive)
    else:
        vi_handle = ValueIteration(game=trans_sys, competitive=competitive)
    vi_handle.solve(debug=debug, plot=plot)
    print("******************************************************************************************************")
    print("Winning strategy exists") if vi_handle.is_winning() else print("No Winning strategy exists")
    print("******************************************************************************************************")


def play_cooperative_game(trans_sys: Union[FiniteTransSys, TwoPlayerGraph],
                          debug: bool = False,
                          plot: bool = False):
    coop_handle = CooperativeGame(game=trans_sys, debug=debug, extract_strategy=True)
    coop_handle.reachability_solver()
    coop_handle.print_winning_region()
    coop_handle.print_winning_strategies()
    
    if plot:
        coop_handle.plot_graph(with_strategy=True)


def play_safety_game(trans_sys: TwoPlayerGraph, debug: bool = False, plot: bool = False):
    """
     A method to compute safe states from Sys player
    """
    assert isinstance(trans_sys, TwoPlayerGraph), "Make sure the graph is an instance of TwoPlayerGraph class for Best effort experimental code."
    safety_handle = SafetyGame(game=trans_sys, target_states= set(["v0", "v1", "v4", "v5", "v8", "v9", "v10", "v13"]),debug=debug)
    # safety_handle = SafetyGame(game=trans_sys, target_states= set(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]),debug=debug)
    safety_handle.reachability_solver()

    safety_handle.print_winning_region()
    safety_handle.print_winning_strategies()
    
    if plot:
        safety_handle.plot_graph(with_strategy=True)



def play_qual_be_synthesis_game(trans_sys: TwoPlayerGraph, debug: bool = False, plot: bool = False, print_states: bool = False):
    """
     A method to compute Qualitative Best effort strategies for the system player
    """
    assert isinstance(trans_sys, TwoPlayerGraph), "Make sure the graph is an instance of TwoPlayerGraph class for Best effort experimental code."
    be_handle = QualitativeBestEffortReachSyn(game=trans_sys, debug=debug)
    be_handle.compute_best_effort_strategies(plot=plot)
    be_handle.get_losing_region(print_states=print_states)
    be_handle.get_pending_region(print_states=print_states)
    be_handle.get_winning_region(print_states=print_states)


def play_quant_be_synthesis_game(trans_sys: TwoPlayerGraph, debug: bool = False, plot: bool = False, print_states: bool = False):
    """
     A method to compute Quantitative Best effort strategies for the system player
    """
    assert isinstance(trans_sys, TwoPlayerGraph), "Make sure the graph is an instance of TwoPlayerGraph class for Best effort experimental code."
    be_handle = QuantitativeBestEffortReachSyn(game=trans_sys, debug=debug)
    be_handle.compute_best_effort_strategies(plot=plot)
    be_handle.get_losing_region(print_states=print_states)
    be_handle.get_pending_region(print_states=print_states)
    be_handle.get_winning_region(print_states=print_states)
    
    # print be strategy dictionary for sanity checking
    for state, succ_states in be_handle.sys_best_effort_str.items():
        print(f"Strategy from {state} is {succ_states}")
    

def play_quant_hopeful_admissbile_synthesis_game(trans_sys: TwoPlayerGraph, debug: bool = False, plot: bool = False, print_states: bool = False):
    """
     A method to compute Quantitative Best effort strategies for the system player
    """
    assert isinstance(trans_sys, TwoPlayerGraph), "Make sure the graph is an instance of TwoPlayerGraph class for Best effort experimental code."
    be_handle = QuantitativeHopefullAdmissibleReachSyn(game=trans_sys, debug=debug)
    be_handle.compute_best_effort_strategies(plot=plot)
    
    # print admissible strategy dictionary for sanity checking
    for state, succ_states in be_handle.sys_best_effort_str.items():
        print(f"Strategy from {state} is {succ_states}")


def play_quant_admissbile_synthesis_game(trans_sys: TwoPlayerGraph, debug: bool = False, plot: bool = False):
    """
     A method to compute Quantitative Admissible strategies for the system player - AAAI 25
    """
    assert isinstance(trans_sys, TwoPlayerGraph), "Make sure the graph is an instance of TwoPlayerGraph class for Best effort experimental code."
    be_handle = QuantitativeGoUAdmissible(game=trans_sys, debug=debug, budget=10)
    be_handle.compute_adm_strategies(plot=plot, plot_transducer=True, compute_str=False)

    
    # be_handle = QuantitativeNaiveAdmissible(game=trans_sys, debug=debug, budget=10)
    # be_handle.compute_adm_strategies(plot=plot)
    
    # print admissible strategy dictionary for sanity checking
    if debug: 
        for state, succ_states in be_handle.sys_best_effort_str.items():
            print(f"Strategy from {state} is {succ_states}")


def play_quant_admissbile_winning_synthesis_game(trans_sys: TwoPlayerGraph, debug: bool = False, plot: bool = False):
    """
     A method to compute Quantitative Admissible Winning strategies for the system player - AAAI 25
    """
    assert isinstance(trans_sys, TwoPlayerGraph), "Make sure the graph is an instance of TwoPlayerGraph class for Best effort experimental code."
    be_handle = QuantitativeGoUAdmissibleWinning(game=trans_sys, debug=debug, budget=10)
    be_handle.compute_adm_strategies(plot=plot, plot_transducer=True, compute_str=False)

    
    # print admissible strategy dictionary for sanity checking
    if debug: 
        for state, succ_states in be_handle.sys_best_effort_str.items():
            print(f"Strategy from {state} is {succ_states}")


def play_quant_refined_admissbile_synthesis_game(trans_sys: TwoPlayerGraph, debug: bool = False, plot: bool = False):
    """
     A method to compute Quantitative refined Admissible strategies for the system player - ICRA 25
    """
    assert isinstance(trans_sys, TwoPlayerGraph), "Make sure the graph is an instance of TwoPlayerGraph class for Best effort experimental code."
    raise NotImplementedError


def finite_reg_minimizing_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph]):
    """
    A new regret computation method. Assumption: The weights on the graph represent costs and are hence non-negative.
    Sys player is minimizing its cumulative cost while the env player is trying to maximize the cumulative cost.

    Steps:
        1. Add an auxiliary tmp_accp state from the accepting state and the trap state. The edge weight is 0 and W_bar
        respectively. W_bar is equal to : (|V| - 1) x W where W is the max absolute weight
        2. Compute the best competitive value and best alternate value for each strategy i.e edge for sys node.
        3. Reg = competitive - cooperative(w')
        4. On this reg graph we then play competitive game i.e Sys: Min player and Env: Max player
        5. Map back these strategies to the original game.

    :param trans_sys:
    :return:
    """
    # build an instance of strategy minimization class
    reg_syn_handle = RegMinStrSyn(trans_sys)

    reg_syn_handle.edge_weighted_arena_finite_reg_solver(purge_states=True,
                                                        plot=False)


def construct_ltlf_dfa():
    """
    A method to construct the LTLf DFA for the given LTLf formula
    """
    
    dfa_handle = graph_factory.get('LTLfDFA',
                                    graph_name="ltlf_automaton",
                                    config_yaml="config/ltltf_automaton",
                                    save_flag=True,
                                    ltlf="!d U g",
                                    plot=True)
    sys.exit(-1)


if __name__ == "__main__":

    # define some constants
    EPSILON = 0  # 0 - the best strategy (for human too) and 1 - Completely random
    IROS_FLAG: bool = False
    ENERGY_BOUND = 30
    ALLOWED_HUMAN_INTERVENTIONS = 2

    # some constants related to computation
    finite: bool = True
    go_fast: bool = True

    # some constants that allow for appr _instance creations
    three_state_ts: bool = False
    five_state_ts: bool = False
    variant_1_paper: bool = False
    target_weighted_arena: bool = False
    two_player_arena: bool = True
    check_ltlf_dfa: bool = False

    # solver to call
    qual_BE_synthesis: bool = False
    quant_BE_synthesis: bool = False
    quant_hopeful_admissibile_synthesis: bool = False
    quant_naive_adm: bool = False # AAAI 25
    quant_adm_winning: bool = False  # AAAI 25
    quant_refined_adm_: bool = False # ICRA 25
    finite_reg_synthesis: bool = False
    infinte_reg_synthesis: bool = False
    adversarial_game: bool = False
    iros_str_synthesis: bool = False
    min_max_game: bool = False
    play_coop_game: bool = False
    safety_game: bool = True

    # build the graph G on which we will compute the regret minimizing strategy
    if three_state_ts:
        three_state_ts_instance = ThreeStateExample(_finite=finite,
                                                    _plot_ts=True,
                                                    _plot_dfa=True,
                                                    _plot_prod=True)
        trans_sys = three_state_ts_instance.product_automaton

    elif five_state_ts:
        five_state_ts = FiveStateExample(_finite=finite,
                                         _plot_ts=False,
                                         _plot_dfa=False,
                                         _plot_prod=False)
        trans_sys = five_state_ts.product_automaton

    elif variant_1_paper:
        variant_1_instance = VariantOneGraph(_finite=finite,
                                             _plot_prod=False)
        trans_sys = variant_1_instance.product_automaton

    elif target_weighted_arena:
        twa_graph = EdgeWeightedArena(_graph_type="ewa",
                                      _plot_prod=True)
        trans_sys = twa_graph.product_automaton
    
    elif check_ltlf_dfa:
        construct_ltlf_dfa()
        sys.exit(-1)

    elif two_player_arena:
        # 4 state example
        # two_player_graph = four_state_BE_example(add_weights=True, plot=False)

        # 8 state example
        # two_player_graph = eight_state_BE_example(add_weights=True, plot=False)

        # Example 2 from Appendix
        # two_player_graph = example_two_BE_example(add_weights=True, plot=False)

        # Example 3 from Appendix
        # two_player_graph = example_three_BE_example(add_weights=True, plot=False)

        # toy adversarial game graph
        # two_player_graph = adversarial_game_toy_example(plot=True)

        # toy admissibility game graph 1
        # two_player_graph = admissibility_game_toy_example_1(plot=False)

        # toy admissibility game graph 2
        # two_player_graph = admissibility_game_toy_example_2(plot=False)

        # toy admissibility game graph 3
        two_player_graph = admissibility_game_toy_example_3(plot=False)

        trans_sys = two_player_graph
        # sys.exit(-1)

    else:
        warnings.warn("Please ensure at-least one of the flags is True")
        sys.exit(-1)

    print(f"No. of nodes in the product graph is :{len(trans_sys._graph.nodes())}")
    print(f"No. of edges in the product graph is :{len(trans_sys._graph.edges())}")

    if finite_reg_synthesis:
        finite_reg_minimizing_str(trans_sys)
    
    elif adversarial_game:
        compute_winning_str(trans_sys, debug=True, plot=True, print_winning_regions=True, print_str=True)
    
    elif iros_str_synthesis:
        compute_bounded_winning_str(trans_sys, energy_bound=ENERGY_BOUND, debug=False, print_str=False)
    
    elif min_max_game:
        play_min_max_game(trans_sys=trans_sys, debug=True, plot=True, permissive_strategies=True, competitive=True)
    
    elif play_coop_game:
        play_cooperative_game(trans_sys=trans_sys, debug=True, plot=True)
    
    elif safety_game:
        play_safety_game(trans_sys=trans_sys, debug=True, plot=True)

    elif qual_BE_synthesis:
        play_qual_be_synthesis_game(trans_sys=trans_sys, debug=True, plot=True, print_states=True)

    elif quant_BE_synthesis:
        play_quant_be_synthesis_game(trans_sys=trans_sys, debug=True, plot=True, print_states=True)
    
    elif quant_hopeful_admissibile_synthesis:
        play_quant_hopeful_admissbile_synthesis_game(trans_sys=trans_sys, debug=True, plot=True, print_states=True)
    
    elif quant_naive_adm:
        play_quant_admissbile_synthesis_game(trans_sys=trans_sys, debug=False, plot=True)
    
    elif quant_adm_winning:
        play_quant_admissbile_winning_synthesis_game(trans_sys=trans_sys, debug=False, plot=False)
    
    elif quant_refined_adm_:
        play_quant_refined_admissbile_synthesis_game(trans_sys=trans_sys, debug=False, plot=False)

    else:
        warnings.warn("Please make sure that you select at-least one solver.")
        sys.exit(-1)
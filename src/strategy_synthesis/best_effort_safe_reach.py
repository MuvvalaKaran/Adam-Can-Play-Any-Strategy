import copy
import warnings

from collections import defaultdict
from typing import Dict, Set, Union, List, Optional

from ..graph import TwoPlayerGraph

from .best_effort_syn import QualitativeBestEffortReachSyn, QualitativeBestEffortSafetySyn
from .best_effort_syn import QuantitativeBestEffortReachSyn, QuantitativeBestEffortSafetySyn


class QualitativeSafeReachBestEffort(QualitativeBestEffortReachSyn):
    """
     A class that implements  my proposed algorithm for IJCAI 24 with Quantitative objectives. The algorithm is as follows:

      1. Compute Losing, Pending and Winning region.
      2. In Winning region compute Winning strategy - BE reachability
      3. In Winning + Pending region compute BE Safety
      4. In Pending region compute BE Reachability game with objective of reaching the Winning region
      5. Merge strategies
    """

    def __init__(self, game: TwoPlayerGraph, debug: bool = False) -> None:
        super().__init__(game, debug)
    

    def compute_best_effort_strategies(self, debug: bool = False, plot: bool = False):
        """
         The main method that implements the safe reach best effort synthesis approach.
        """

        sys_best_effort_pending_str: Dict[str, Set[str]] = defaultdict(lambda: set({}))

        # get winning strategies
        self.compute_winning_strategies()

        # get cooperative winning region
        self.compute_cooperative_winning_strategy()

        # remove env states from the winning region and cooperative winning region. 
        self._winning_region = [ws for ws in self._winning_region if self._game.get_state_w_attribute(ws, 'player') == 'eve']
        # self._coop_winning_region = [cs for cs in self._coop_winning_region if self._game.get_state_w_attribute(cs, 'player') == 'eve']
        self._coop_winning_region = [cs for cs in self._coop_winning_region]
        self._pending_region = set(self.coop_winning_region).difference(set(self.winning_region))

        # now compute BE Safety strategies
        safety_be_handle = QualitativeBestEffortSafetySyn(game=self.game, target_states=self._coop_winning_region, debug=True)
        safety_be_handle.compute_best_effort_safety_strategies(plot=True)

        if debug:
            print("BE Safe Str in Pending + Winning Region: ", safety_be_handle.sys_best_effort_str)
        
        be_reach_win_trans_sys = copy.deepcopy(self.game)
        # TODO: Should this be the winning_region such that there exists winning stratgey from accepting states or not?
        be_reach_win_trans_sys.add_accepting_states_from(self._winning_region)

        # Finally, compute reachability strategies from the pending region with Winning region as reacability objective
        be_handle = QualitativeBestEffortReachSyn(game=be_reach_win_trans_sys, debug=False)
        # TODO: We need to code up Qualitative Permissive Reachability code.
        be_handle.compute_best_effort_strategies(plot=False)
        
        if debug:
            print("BE Reach Str in Pending + Winning to winning region: ", be_handle.sys_best_effort_str)
        
        # for pending states (ps)
        for ps in self.pending_region:
            if self.game.get_state_w_attribute(ps, 'player') == 'eve':
                try:
                    safereach_str = set(be_handle.sys_best_effort_str[ps]).intersection(safety_be_handle.sys_best_effort_str[ps])
                    if safereach_str:
                        sys_best_effort_pending_str[ps].update(safereach_str)
                    else:
                        sys_best_effort_pending_str[ps].update(set(be_handle.sys_best_effort_str[ps]))

                except KeyError:
                    warnings.warn(f"Something went wrog during Best Effort Synthesis in Pending Region! \
                                state {ps} does not exists in BE Safety and BE Reachability strategy dictionary!")
        
        # override the sys_coop_winning_str dictionary that computed in compute_cooperative_winning_strategy() method above
        self._sys_coop_winning_str = sys_best_effort_pending_str

        # for states that belong to the losing region, we can play any strategy
        _sys_losing_str = {state: list(self.game._graph.successors(state)) for state in self.get_losing_region() if self.game.get_state_w_attribute(state, 'player') == 'eve'}
        
        self._sys_best_effort_str: Dict[str, Set[str]] = {**self.sys_winning_str, **self._sys_coop_winning_str, **_sys_losing_str}

        if plot:
            self.add_str_flag()
            self.game.plot_graph()
        
        if debug:
            print("Done Computing Strategies.")


class QuantitativeSafeReachBestEffort(QuantitativeBestEffortReachSyn):
    """
    This methods implements my proposed algorithm for IJCAI 24 with Quantitative objectives. The algorithm is as follows:

     1. Compute Losing, Pending and Winning region.
     2. In Winning region compute Winning strategy - BE reachability
     3. In Winning + Pending region compute BE Safety
     4. In Pending region compute BE Reachability game with objective of reaching the Winning region
     5. Merge strategies
    
     The algorithm is same as the Qualitative one.
    """

    def __init__(self, game: TwoPlayerGraph, debug: bool = False) -> None:
        super().__init__(game, debug)
    

    def compute_best_effort_strategies(self, debug: bool = False, plot: bool = False):
        """
         The main method that implements the safe reach best effort synthesis approach.
        """

        sys_best_effort_pending_str: Dict[str, Set[str]] = defaultdict(lambda: set({}))

        # get winning strategies
        self.compute_winning_strategies(permissive=False)

        # get cooperative winning strategies
        self.compute_cooperative_winning_strategy()

        # remove env states from the winning region and cooperative winning region. 
        self._winning_region = [ws for ws in self._winning_region if self.game.get_state_w_attribute(ws, 'player') == 'eve']
        self._coop_winning_region = [cs for cs in  self._coop_winning_region if self.game.get_state_w_attribute(cs, 'player') == 'eve']
        self._pending_region = set(self._coop_winning_region).difference(set(self._winning_region))


        if debug:
            print("Winning Region: ", self.winning_region)
            print("Pending Region: ", self.pending_region)
        

        # now compute BE Safety strategies
        safety_be_handle = QuantitativeBestEffortSafetySyn(game=self.game, target_states=self._coop_winning_region, debug=False)
        safety_be_handle.compute_best_effort_safety_strategies(plot=True)

        if debug:
            print("BE Safe Str in Pending + Winning Region: ", safety_be_handle.sys_best_effort_str)

        be_reach_win_trans_sys = copy.deepcopy(self.game)
        # TODO: Should this be the winning_region such that there exists winning stratgey from accepting states or not? 
        be_reach_win_trans_sys.add_accepting_states_from(self._winning_region)

        # Finally, compute reachability strategies from the pending region with Winning region as reacability objective
        be_handle = QuantitativeBestEffortReachSyn(game=be_reach_win_trans_sys, debug=False)
        be_handle.compute_best_effort_strategies(plot=True)
        
        if debug:
            print("BE Reach Str to winning region: ", be_handle.sys_best_effort_str)
        

        # for pending states (ps)
        for ps in self.pending_region:
            try:
                safereach_str = set(be_handle.sys_best_effort_str[ps]).intersection(safety_be_handle.sys_best_effort_str[ps])
                if safereach_str:
                    sys_best_effort_pending_str[ps].update(safereach_str)
                else:
                    sys_best_effort_pending_str[ps].update(set(be_handle.sys_best_effort_str[ps]))

            except KeyError:
                warnings.warn(f"SOmething went wrog during Best Effort Synthesis in Pending Region! \
                            state {ps} does not exists in BE Safety and BE Reachability strategy dictionary!")
        
        # override the sys_coop_winning_str dictionary that computed in compute_cooperative_winning_strategy() method above
        self._sys_coop_winning_str = sys_best_effort_pending_str

        # for states that belong to the losing region, we can play any strategy
        _sys_losing_str = {state: list(self.game._graph.successors(state)) for state in self.get_losing_region() if self.game.get_state_w_attribute(state, 'player') == 'eve'}
        
        self._sys_best_effort_str: Dict[str, Set[str]] = {**self.sys_winning_str, **self._sys_coop_winning_str, **_sys_losing_str}

        if plot:
            self.add_str_flag()
            self.game.plot_graph()
        
        if debug:
            print("Done Computing Strategies.")






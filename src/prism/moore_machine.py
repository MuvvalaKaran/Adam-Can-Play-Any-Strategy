import random
from collections import defaultdict
from scipy.stats import rv_discrete, uniform
from typing import Union, List, Dict, Tuple

NodeName = Union[str, Tuple[str]]
ActionName = str


class MooreMachine:

    """Initial State: s \in S"""
    _init_state: int = None

    """Initial Distribution: S -> D(M)"""
    _init_dist: rv_discrete = None

    """Next Function: S x M -> A"""
    #               state, memory, action
    _next_dict: Dict[int, Dict[int, int]] = None

    """Memory Update States Function: S x M x A -> D(M)"""
    #                       state, memory, action, distribution(memory)
    _update_state_dict: Dict[int, Dict[int, Dict[int, rv_discrete]]] = None

    """Memory Update Moves Function: S x A x M -> S x D(M)"""
    #                       state, action, memory, next_state, distribution(memory)
    _update_move_dict: Dict[int, Dict[int, Dict[int, Dict[int, rv_discrete]]]] = None

    """Current State s"""
    _curr_state: int = None

    """Current Action a"""
    _curr_action: int = None

    """Current Memory m"""
    _curr_memory: int = None

    """On-Move Memory o_m"""
    _onmove_memory: int = None

    def __init__(self,
        init_state: int = None,
        init_distribution: rv_discrete = None,
        next_dict: Dict[int, Dict[int, int]] = None,
        update_state_dict: Dict[int, Dict[int, Dict[int, rv_discrete]]] = None,
        update_move_dict: Dict[int, Dict[int, Dict[int, Dict[int, rv_discrete]]]] = None):

        self._init_state = init_state
        self._init_distribution = init_distribution
        self._next_dict = next_dict
        self._update_state_dict = update_state_dict
        self._update_move_dict = update_move_dict

        self._check_initialized()

        self._curr_state = self._init_state
        self._curr_memory = self._init_distribution.rvs(size=1)[0]

    def get_memories(self, curr_state: int) -> List[int]:

        self._check_initialized()

        if curr_state in self._update_state_dict:
            curr_memories = set(self._update_state_dict[curr_state].keys())
            return curr_memories

        raise ValueError(f'Invalid state and memory ({curr_state})')

    def take_action(self, curr_state: int = None, curr_memory: int = None) -> int:
        """Take an aciton and return the next state"""
        if curr_state is None:
            curr_state = self._curr_state

        if curr_memory is None:
            curr_memory = self._curr_memory

        # First, Choose an action
        curr_action = self.sample_action(curr_state, curr_memory)

        # Update Memory
        onmove_memory = self.sample_on_move_memory(curr_state, curr_memory, curr_action)

        # Take a transition
        next_state = self.get_next_states(curr_state, curr_action, onmove_memory)
        next_memory = self.sample_next_memory(curr_state, curr_action, onmove_memory)

        self._curr_state = next_state
        self._curr_memory = next_memory

        return next_state

    def sample_action(self, curr_state: int, curr_memory: int) -> int:
        return self._sample_action(curr_state, curr_memory)

    def __sample_action(self, curr_state: int, curr_memory: int) -> int:
        actions = self._get_actions(curr_state, curr_memory)

        if isinstance(actions, List):
            # Adam, Env
            return random.choice(actions)
        else:
            # Eve, Sys
            return actions

    def get_actions(self, curr_state: int, curr_memory: int) -> List[int]:
        return self.__get_actions(curr_state, curr_memory)

    def __get_actions(self, curr_state: int, curr_memory: int) -> List[int]:

        self._check_initialized()

        # Eve's State
        if curr_state in self._next_dict:
            if curr_memory in self._next_dict[curr_state]:
                return [self._next_dict[curr_state][curr_memory].rvs(size=1)[0]]
            else:
                raise ValueError(f'Memory {curr_memory} is not in the next function')

        # Adam's State
        if curr_state in self._update_state_dict:
            if curr_memory in self._update_state_dict[curr_state]:
                next_actions = set(self._update_state_dict[curr_state][curr_memory].keys())
                return next_actions

        raise ValueError(f'Invalid state and memory ({curr_state}, {curr_memory})')

    def sample_on_move_memory(self, curr_state: int, curr_memory: int, curr_action: int) -> int:
        return self.__sample_on_move_memory(curr_state, curr_memory, curr_action)

    def __sample_on_move_memory(self, curr_state: int, curr_memory: int, curr_action: int) -> int:

        self._check_initialized()

        if curr_state in self._update_state_dict:
            if curr_memory in self._update_state_dict[curr_state]:
                if curr_action in self._update_state_dict[curr_state][curr_memory]:
                    return self._update_state_dict[curr_state][curr_memory].rvs(sample=1)[0]

        msg = f'Invalid state, memory, action = ({curr_state}, {curr_memory}, {curr_action})'
        raise ValueError(msg)

    def get_onmove_memories(self, curr_state: int, curr_memory: int, curr_action: int) -> List[int]:
        return self.__get_onmove_memories()

    def __get_onmove_memories(self, curr_state: int, curr_memory: int, curr_action: int) -> List[int]:

        self._check_initialized()

        if curr_state in self._update_state_dict:
            if curr_memory in self._update_state_dict[curr_state]:
                if curr_action in self._update_state_dict[curr_state][curr_memory]:
                    return self._update_state_dict[curr_state][curr_memory][curr_action].xk

        msg = f'Invalid state, memory, action = ({curr_state}, {curr_memory}, {curr_action})'
        raise ValueError(msg)

    # TODO: Currently, we assume next state is deterministically transitioned from s,a,m
    def get_next_state(self, curr_state: int, curr_action: int, onmove_memory: int) -> int:
        return self.__get_next_state(curr_state, curr_action, onmove_memory)

    def __get_next_state(self, curr_state: int, curr_action: int, onmove_memory: int) -> int:

        self._check_initialized()

        if curr_state in self._update_move_dict:
            if curr_action in self._update_move_dict[curr_state]:
                if onmove_memory in self._update_move_dict[curr_state][curr_action]:
                    next_state = list(self._update_move_dict[curr_state][curr_action][onmove_memory].keys())[0]
                    return next_state

        msg = f'Invalid state, action, memory ({curr_state}, {curr_action}, {onmove_memory})'
        raise ValueError(msg)

    def sample_next_memory(self, curr_state: int, curr_action: int, onmove_memory: int) -> int:
        return self.__sample_next_memory(curr_state, curr_action, onmove_memory)

    def __sample_next_memory(self, curr_state: int, curr_action: int, onmove_memory: int) -> int:

        self._check_initialized()

        next_state = self.__get_next_state(curr_state, curr_action, onmove_memory)

        if next_state in self._update_move_dict[curr_state][curr_action][onmove_memory]:
            return self._update_move_dict[curr_state][curr_action][onmove_memory][next_state].rvs(sample=1)[0]

        msg = f'Invalid state, action, memory ({curr_state}, {curr_action}, {onmove_memory})'
        raise ValueError(msg)

    def get_next_memories(self, curr_state: int, curr_action: int,
                          onmove_memory: int, next_state: int) -> List[int]:
        return self.__get_next_memories(curr_state, curr_action, onmove_memory, next_state)

    def __get_next_memories(self, curr_state: int, curr_action: int,
                          onmove_memory: int, next_state: int) -> List[int]:

        self._check_initialized()

        next_state = self.__get_next_state(curr_state, curr_action, onmove_memory)

        if next_state in self._update_move_dict[curr_state][curr_action][onmove_memory]:
            next_memories = self._update_move_dict[curr_state][curr_action][onmove_memory][next_state].xk
            return next_memories

        msg = f'Invalid state, action, memory, next_state ({curr_state}, {curr_action}, {onmove_memory}, {next_state})'
        raise ValueError(msg)

    def get_transitions(self) -> Dict[int, List[int]]:

        states = set(self._update_move_dict.keys())
        transitions: Dict[int, List[int]] = defaultdict(lambda: [])

        for state in states:

            memories = self.get_memories(state)
            for memory in memories:

                actions = self.__get_actions(state, memory)
                for action in actions:

                    onmove_memories = self.__get_onmove_memories(state, memory, action)
                    for onmove_memory in onmove_memories:

                        next_state = self.__get_next_state(state, action, onmove_memory)
                        transition = {'next_node': next_state, 'action': action}

                        transitions[state].append(transition)

        return transitions

    def _check_initialized(self):
        if not self.initialized:
            raise Exception('Not initialized')

    @property
    def initialized(self):
        if None in [self._init_state,
            self._init_distribution,
            self._next_dict,
            self._update_state_dict,
            self._update_move_dict]:
            return False
        return True

    @property
    def init_state(self) -> int:
        self._check_initialized()
        return self._init_state

    @property
    def curr_state(self) -> int:
        return self._curr_state

    @property
    def curr_memory(self) -> int:
        return self._curr_memory

    @property
    def curr_action(self) -> int:
        if self._curr_action is None:
            raise Exception('Current Action not initialized. Please take an action first')
        return self._curr_action


class PrismMooreMachine(MooreMachine):

    """Prism STA state (strategy state) to prism model state"""
    _sta_to_prism_state_map: Dict[int, int] = None

    """Prism model state to game state"""
    _prism_to_game_map: Dict[int, NodeName] = None

    """State & Action idx to Action String"""
    _prism_action_idx_to_game_map: Dict[int, Dict[int, ActionName]] = None

    def __init__(self,
        sta_to_prism_state_map: Dict[int, int] = None,
        prism_to_game_map: Dict[int, NodeName] = None,
        prism_action_idx_to_game_map: Dict[int, Dict[int, ActionName]] = None,
        init_state: int = None,
        init_distribution: rv_discrete = None,
        next_dict: Dict[int, Dict[int, int]] = None,
        update_state_dict: Dict[int, Dict[int, Dict[int, rv_discrete]]] = None,
        update_move_dict: Dict[int, Dict[int, Dict[int, Dict[int, rv_discrete]]]] = None):
        super().__init__(init_state, init_distribution, next_dict,
            update_state_dict, update_move_dict)

        self._sta_to_prism_state_map = sta_to_prism_state_map
        self._prism_to_game_map = prism_to_game_map
        self._prism_action_idx_to_game_map = prism_action_idx_to_game_map

        self._check_initialized()

        self._prism_to_sta_state_map = dict(zip(self._sta_to_prism_state_map.values(),
                                                self._sta_to_prism_state_map.keys()))
        self._game_to_prism_map = dict(zip(self._prism_to_game_map.values(),
                                           self._prism_to_game_map.keys()))
        # self._game_to_prism_action_idx_map = dict(zip(self._prism_action_idx_to_game_map.values(),
        #                                               self._prism_action_idx_to_game_map.keys()))

    def _sta_to_game(self, sta_state: int) -> NodeName:
        prism_state = self._sta_to_prism_state_map[sta_state]
        return self._prism_to_game_map[prism_state]

    def _game_to_sta(self, game_state: NodeName) -> int:
        prism_state = self._game_to_prism_map[game_state]
        return self._prism_to_sta_state_map[prism_state]

    def init_state(self) -> NodeName:
        state_idx_sta = super().init_state()
        return self._sta_to_game(state_idx_sta)

    def get_transitions(self, return_dict: bool = True, return_edges: bool = False) \
        -> Union[Dict[int, List[Dict[NodeName, NodeName]]], List[Dict[NodeName, NodeName]]]:

        if sum([return_dict, return_edges]) != 1:
            raise ValueError('Select either "return_dict" or "return_edges"')

        transitions = super().get_transitions()

        if return_dict:
            game_transitions = defaultdict(lambda: [])

            for sta_curr_state, transition in transitions.items():
                for transition in transition:
                    sta_next_state = transition['next_node']
                    sta_action = transition['action']
                    game_curr_state = self._sta_to_game(sta_curr_state)
                    game_next_state = self._sta_to_game(sta_next_state)
                    prism_action = self._prism_action_idx_to_game_map[sta_curr_state][sta_action]

                    transition = {'next_node': game_next_state, 'action': prism_action}
                    game_transitions[game_curr_state].append(transition)

            return game_transitions

        elif return_edges:
            game_transitions = []

            for sta_curr_state, sta_next_states in transitions.items():
                for sta_next_state in sta_next_states:
                    game_curr_state = self._sta_to_game(sta_curr_state)
                    game_next_state = self._sta_to_game(sta_next_state)
                    game_transitions.append((game_next_state, game_next_state))

            return game_transitions

    # @property
    # def initialized(self):
    #     if not super().initialized:
    #         return False

    #     if None in [
    #         self._sta_to_prism_state_map,
    #         self._prism_to_game_map,
    #         self._prism_action_idx_to_game_map]:
    #         return False

    #     return True

    @property
    def curr_state(self) -> NodeName:
        return self._sta_to_game(self._curr_state)

    @property
    def curr_action(self) -> ActionName:
        if self._curr_action is None:
            raise Exception('Current Action not initialized. Please take an action first')

        return self._prism_action_idx_to_game_map[self.curr_state][self._curr_action]

    def sample_action(self, curr_state: NodeName, curr_memory: int) -> ActionName:
        # Assume curr_state is of type NodeName
        sta_curr_state = self._game_to_sta(curr_state)

        action_idx = super().sample_action(sta_curr_state, curr_memory)
        return self._prism_action_idx_to_game_map[sta_curr_state][action_idx]

    def get_actions(self, curr_state: NodeName, curr_memory: int) -> List[ActionName]:
        sta_curr_state = self._game_to_sta(curr_state)

        action_idxs = super().get_actions(sta_curr_state, curr_memory)
        return [self._prism_action_idx_to_game_map[sta_curr_state][action_idx] for action_idx in action_idxs]

    def get_next_state(self, curr_state: NodeName, curr_action: ActionName,
                       onmove_memory: int) -> NodeName:
        sta_curr_state = self._game_to_sta(curr_state)

        sta_curr_action = self._prism_action_idx_to_game_map[sta_curr_state].index(curr_action)

        state_idx = super().get_next_state(sta_curr_state, sta_curr_action, onmove_memory)
        return self._sta_to_game(state_idx)

from abc import ABC, abstractmethod
from collections import namedtuple


ActionResult = namedtuple('ActionResult', ['probability', 'next_state', 'reward'])


class AbstractDynamics(ABC):
    """Provides an interface for all Dynamics classes to implement."""
    @abstractmethod
    def state_list(self) -> list:
        """Returns a list of all legal states, excluding terminal states."""
        pass


    @abstractmethod
    def terminal_state_list(self) -> list:
        """Returns a list of all terminal states."""
        pass

    @abstractmethod
    def legal_actions(self, input_state) -> list:
        """Given an input state, returns a list of actions that can be fed to state_action_results.
        An empty list implies no legal actions are allowed, AKA you've been given a terminal state.
        """
        pass

    @abstractmethod
    def state_action_results(self, input_state, input_action) -> list[ActionResult]:
        """Maps a given input state + action to a list of possible StateReward tuples.
        """
        pass


import matplotlib.pyplot as plt

# Extremely Simple Dynamics for testing:
# An array of ints. Can only move left and right.
# Reward is equal to difference in the states you're
# moving between. Moving off the grid ends the game.
class DumbDynamics(AbstractDynamics):
    grid = []
    def __init__(self, grid):
        self.grid = grid

    def state_list(self):
        return list(range(len(self.grid)))

    def terminal_state_list(self):
        return [-1, len(self.grid)]

    def legal_actions(self, input_state) -> list:
        if input_state < -1 or input_state > len(self.grid):
            raise ValueError(f'{input_state} is not a valid state for grid: {self.grid}')
        if input_state == -1 or input_state == len(self.grid):
            return [] # Signals Terminal State

        return [-1, 1] # TODO: this should be named 'left', 'right', probably? Maybe make an enum

    def state_action_results(self, input_state, input_action) -> list[ActionResult]:
        if input_state < 0 or input_state >= len(self.grid):
            raise ValueError(f'{input_state}, {input_action} is not a valid state action pair for grid: {self.grid}')
        if input_action != -1 and input_action != 1:
            raise ValueError(f'{input_action} is not a valid action, try -1 or 1.')
        next_state = input_state + input_action
        if next_state == -1 or next_state == len(self.grid):
            reward = 0
        else:
            reward = self.grid[next_state] - self.grid[input_state]
        return [ActionResult(
            probability=1,
            next_state=next_state,
            reward=reward
        )]

    def visualize_policy(self, policy: dict, name: str=None) -> None:
        """Prints a picture of the policy given"""
        to_print = []
        for i in range(len(self.grid)):
            if policy[i] == 1:
                to_print.append('->')
            else:
                to_print.append('<-')
        if name is not None:
            print(f'Policy {name}: {to_print}')
        else:
            print('Policy :', to_print)

    def visualize_values(self, values: dict, name: str='Values Graph') -> None:
        """Shows a bar chart of the values given."""
        to_chart = [values[i] for i in range(len(self.grid))]
        plt.bar(list(range(len(self.grid))), to_chart)
        plt.title(name)
        plt.show()


# Example 4.2 Jack's Car Rental
from itertools import combinations, product
from numpy.random import poisson
class CarDynamics(AbstractDynamics):
    num_cars = (0, 0)
    verbose = True

    def __init__(self, first_loc_num, second_loc_num):
        self.num_cars = (first_loc_num, second_loc_num)

    def state_list(self) -> list:
        return [c for c in combinations(range(21), 2)]

    def terminal_state_list(self) -> list:
        return []

    def legal_actions(self, input_state: tuple) -> list:
        """No more than 20 cars per site, only 5 cars can be moved

        Action is defined as an integer. 5 corresponds to moving 5 cars
        to the second location. -5 corresponds to moving 5 cars to the
        first location.
        """
        max_movable_to_second = min(min(5, input_state[0]), 20 - input_state[1])
        max_movable_to_first = min(min(5, input_state[1]), 20 - input_state[0])
        return list(range(-1 * max_movable_to_first, max_movable_to_second + 1))

    def state_action_results(self, input_state: tuple, input_action: int) -> list[ActionResult]:
        # Going to need to use the poisson distribution to calculate the probability
        # of n customers requesting/returning cars for n <= 20?
        pass

    def visualize_policy(self, policy: dict):
        to_print = [[0 for _ in range(21)] for _ in range(21)]
        for i in range(21):
            for j in range(21):
                to_print[i][j] = policy[(i, j)]
        for row in to_print:
            print(row)


class PolicyIteration:
    """Implements Policy Iteration as described in Sutton 4.3"""
    # maps states to values
    values = {}
    # maps states to actions
    policy = {}

    gamma = 0.9
    theta = 0.1

    state_list = None
    terminal_state_list = None
    dynamics = None

    def __init__(self, dynamics: AbstractDynamics, gamma=None, theta=None):
        """Initializes Policy Iteration alg as described in Sutton 4.3.

        Arbitrary initialization strategy sets value of all states
        (terminal and non-terminal) to 0 and sets the action for each
        state to just the "first" legal action, as returned by the
        dynamics.logal_actions() function.

        params:
        dynamics: Should be a concrete instantiation of the AbstractDynamics class.
        gamma: float between 0 and 1, determines how highly to prioritize future values
        theta: float > 0, determines when to stop policy iteration
        """
        self.dynamics = dynamics
        self.state_list = dynamics.state_list()
        self.terminal_state_list = dynamics.terminal_state_list()
        if gamma is not None:
            self.gamma = gamma
        if theta is not None:
            self.theta = theta
        for state in self.state_list:
            # Arbitrary initialization, in future can test different
            # possible inits to test which one performs best.
            self.values[state] = 0
            legal_actions = self.dynamics.legal_actions(state)
            self.policy[state] = legal_actions[0] # Simply select the first action
        for terminal_state in self.terminal_state_list:
            self.values[terminal_state] = 0

    def _state_action_valuation(self, state, action):
        """Calculates value for a given state action pair."""
        action_results = self.dynamics.state_action_results(state, action)
        return sum(
            ar.probability * (ar.reward + self.gamma * self.values[ar.next_state])
            for ar in action_results
        )


    def policy_evaluation(self):
        """Performs Policy Evaluation as described in Sutton 4.3."""
        delta = self.theta + 1
        while delta > self.theta:
            delta = 0
            for state in self.state_list:
                v = self.values[state]
                self.values[state] = self._state_action_valuation(state, self.policy[state])
                delta = max(delta, abs(v - self.values[state]))

    def policy_improvement(self):
        """Performs policy improvement as described in Sutton 4.3."""
        policy_stable = True
        for state in self.state_list:
            old_action = self.policy[state]
            # Find argmax for action given the current value function
            best_val = -100000 # This is so bad, I'm sure there's an argmax
            # function out there that I can use.
            best_action = None
            legal_actions = self.dynamics.legal_actions(state)
            for action in legal_actions:
                current_val = self._state_action_valuation(state, action)
                if current_val > best_val:
                    best_val, best_action = current_val, action
            if best_action is None:
                raise Exception("You messed up chris. Policy Improvement Failed.")
            self.policy[state] = best_action
            if old_action != self.policy[state]:
                policy_stable = False
        return policy_stable

    def run_alg(self, print_progress=False):
        loops = 0
        while True:
            if print_progress:
                print('loops: ', loops)
                print('values: ', self.values)
                print('policy: ', self.policy)
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            if policy_stable:
                return self.values, self.policy
            loops += 1

def test_dumb_dynamics():
    dyn = DumbDynamics([-1, 0, 1])
    policy_iteration = PolicyIteration(dyn)
    optimal = policy_iteration.run_alg()
    print('values', optimal[0])
    print('policy', optimal[1])
    dyn.visualize_values(optimal[0], name='Values Graph for Dumb Gridworld')
    dyn.visualize_policy(optimal[1], name='dumb')

    dyn = DumbDynamics([1, 0, -1])
    policy_iteration = PolicyIteration(dyn)
    optimal = policy_iteration.run_alg()
    print('values', optimal[0])
    print('policy', optimal[1])
    dyn.visualize_values(optimal[0], name='Values Graph for Invert Dumb Gridworld')
    dyn.visualize_policy(optimal[1], name='invert dumb')

    trick = DumbDynamics([100, 10, 0, 0, 0, 0, 1000])
    policy_iteration = PolicyIteration(trick)
    optimal = policy_iteration.run_alg()
    print('optimal values', optimal[0])
    print('optimal policy', optimal[1])
    trick.visualize_values(optimal[0], name='Values Graph for Trick Gridworld')
    trick.visualize_policy(optimal[1], name='trick')


if __name__ == '__main__':
    # test_dumb_dynamics()
    car_dyn = CarDynamics(0, 0)
    di = {(i, j): (i + j) % 5 for i, j in product(range(21), repeat=2)}
    car_dyn.visualize_policy(di)
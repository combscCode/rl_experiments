from itertools import combinations
from collections import namedtuple
ActionTriple = namedtuple('ActionTriple', ['action', 'next_state', 'reward'])
class PolicyIteration:
    # maps states to values
    values = {}
    # maps states to actions
    policy = {}

    gamma = 0.9
    theta = 0.1
    state_list = None
    action_function = None

    def __init__(self, state_list, action_function, gamma=None, theta=None):
        """Initializes Policy Iteration alg as described in Sutton 4.3 assuming
        action + state pairs only produce one reward + next state pair.
        params:
        state_list: list of all states
        action_function: callable that maps state -> list of all possible
                         future (action, next_state, reward) triples.
                         Note: action used to improve interpretability
                         of what the model is doing. Not explicitly
                         needed to compute optimal policies.
        gamma: float between 0 and 1, determines how highly to prioritize future values
        theta: float > 0, determines when to stop policy iteration
        """
        self.state_list = state_list
        self.action_function = action_function
        if gamma is not None:
            self.gamma = gamma
        if theta is not None:
            self.theta = theta
        for state in state_list:
            # Arbitrary initialization, in future can test different
            # possible inits to test which one performs best.
            self.values[state] = 0
            action_triples = action_function(state)
            if len(action_triples):
                self.policy[state] = action_triples[0].action # Simply select the first action
            else:
                self.policy[state] = None

    def policy_evaluation(self):
        """Performs Policy Evaluation as described in Sutton 4.3 assuming
        action + state pairs only produce one reward + next state pair.
        """
        delta = 500
        while delta > self.theta:
            delta = 0
            for state in self.state_list:
                v = self.values[state]
                action_triples = self.action_function(state)
                found_action = False # used to debug broken code
                for at in action_triples:
                    if at.action == self.policy[state]:
                        found_action = True
                        self.values[state] = at.reward + self.gamma * self.values[at.next_state]
                        break
                if not found_action and len(action_triples):
                    raise Exception('Could not find action in policy evaluation uh oh')
                if found_action:
                    delta = max(delta, abs(v - self.values[state]))

    def policy_improvement(self):
        """Performs policy improvement as described in Sutton 4.3 assuming
        action + state pairs only produce one reward + next state pair."""
        policy_stable = True
        for state in self.state_list:
            old_action = self.policy[state]
            action_triples = self.action_function(state)
            best_val = -100000 # this is bad but I'm being lazy ¯\_(ツ)_/¯
            for at in action_triples:
                if at.reward + self.gamma * self.values[at.next_state] > best_val:
                    self.policy[state] = at.action
                    best_val = at.reward
            if old_action != self.policy[state]:
                policy_stable = False
        return policy_stable

    def run_alg(self):
        loops = 0
        while True:
            print('loops: ', loops)
            print('values: ', self.values)
            print('policy: ', self.policy)
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            if policy_stable:
                return self.values, self.policy
            loops += 1


from abc import ABC, abstractmethod

class AbstractDynamics(ABC):
    """Provides an interface for all Dynamics classes to implement."""
    @abstractmethod
    def state_list():
        pass

    @abstractmethod
    def action_function():
        pass

# Extremely Simple Dynamics for testing:
# An array of ints. Can only move left and right.
# Reward is equal to difference in the states you're
# moving between. Moving off the grid ends the game.
class DumbDynamics(AbstractDynamics):
    grid = []
    def __init__(self, grid):
        self.grid = grid

    def state_list(self):
        return list(range(-1, len(self.grid) + 1))

    def action_function(self, input_state):
        if input_state < -1 or input_state > len(self.grid):
            raise ValueError(f'{input_state} is not a valid state for grid: {self.grid}')
        if input_state == -1 or input_state == len(self.grid):
            return [] # Signals terminal state
        left_triple = ActionTriple(
            -1,
            input_state - 1,
            0 if input_state == 0 else self.grid[input_state - 1] - self.grid[input_state]
        )
        right_triple = ActionTriple(
            1,
            input_state + 1,
            0 if input_state == len(self.grid) - 1 else self.grid[input_state + 1] - self.grid[input_state]
        )
        return [left_triple, right_triple]

if __name__ == '__main__':
    dyn = DumbDynamics([-1, 0, 1])
    policy_iteration = PolicyIteration(dyn.state_list(), dyn.action_function)
    optimal = policy_iteration.run_alg()
    print(optimal)

    trick = DumbDynamics([-10, 100, 10, 5])
    policy_iteration = PolicyIteration(trick.state_list(), trick.action_function)
    optimal = policy_iteration.run_alg()
    print(optimal)

# Example 4.2 Jack's Car Rental
from numpy.random import poisson
class CarDynamics:
    _num_cars = [0, 0]
    verbose = True

    def __init__(self, first_loc_num, second_loc_num):
        self._num_cars[0] = first_loc_num
        self._num_cars[1] = second_loc_num

    def move_cars(self, num_to_move):
        """Perform the nightly action of moving cars from loc1 to loc2. Returns cost"""
        if num_to_move < -5 or num_to_move > 5:
            raise ValueError('You can only move a maximum of 5 cars. -5 <= num_to_move <= 5')
        self._num_cars[0] = min(self._num_cars[0] - num_to_move, 20)
        self._num_cars[1] = min(self.num_cars[1] + num_to_move, 20)
        if self._num_cars[0] < 0 or self._num_cars[1] < 0:
            raise ValueError('You cannot move more cars out of a location than already exist')
        return abs(num_to_move) * -2

    def day_cycle_helper(self, loc, rental_requests, returns):
        """Simulate a day of rentals and returns for a given location. Returns num rentals made."""
        rentals_made = min(rental_requests, self._num_cars[loc])
        self._num_cars[loc] += returns - rentals_made
        return rentals_made

    def day_cycle(self):
        """Simulate a day of rentals and returns. Returns money Jack makes."""
        
        first_requests = poisson(3)
        second_requests = poisson(4)
        first_returns = poisson(3)
        second_returns = poisson(2)

        if self.verbose:
            print('Begin Day Cycle: ', self._num_cars)
            print('First Rental Requests: ', first_requests)
            print('Second Rental Requests: ', second_requests)
            print('First Rental Returns: ', first_returns)
            print('Second Rental Returns: ', second_returns)

        first_loc_profits = self.day_cycle_helper(0, first_requests, first_returns) * 10
        second_loc_profits = self.day_cycle_helper(1, second_requests, second_returns) * 10

        if self.verbose:
            print('End Day Cycle: ', self._num_cars)
            print('profits: ', first_loc_profits + second_loc_profits)
        return first_loc_profits + second_loc_profits

# dynamics = CarDynamics(0, 0)
# for _ in range(10):
#     dynamics.day_cycle()
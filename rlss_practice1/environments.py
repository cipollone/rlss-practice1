"""Environment configurations, maps and transition functions"""

import itertools
import random
import time
from collections import defaultdict
from typing import SupportsFloat, cast

import gymnasium as gym
import numpy as np
from minigrid import envs
from minigrid.core.constants import DIR_TO_VEC
from minigrid.minigrid_env import MiniGridEnv


class Empty(gym.Wrapper):
    """An Empty minigrid environment with explicit transition and reward functions.

    The agent is rewarded upon reaching the goal location.
    For the observation space see `DecodeObservation`.

    Action space:

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |

    """
    # TODO: update observation space and transition matrices in docstring
    # TODO: action failure is inconsistent with the definition of categorical function

    StateT = tuple[int, int, int]
    ActionT = int

    def __init__(self, seed: int, failure=0.0, **kwargs):
        """Initialize.

        seed: random seed
        failure: failure probability of the actions (another action is executed instead).
        size: room side length
        agent_start_pos: tuple with coordinates
        agent_start_dir: north or..
        """
        # Create minigrid env
        self.minigrid = envs.EmptyEnv(highlight=False, **kwargs)
        self.minigrid.action_space = gym.spaces.Discrete(3)
        self.failure = failure

        # Transform appropriately
        env: gym.Env = FailProbability(self.minigrid, failure=failure, seed=seed)
        env = DecodeObservation(env=env)
        env = BinaryReward(env=env)

        # Sizes
        assert isinstance(env.observation_space, gym.spaces.MultiDiscrete)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        obs_space = env.observation_space.nvec
        self.states = list(itertools.product(*(range(obs_space[i]) for i in range(len(obs_space)))))
        self.actions = list(range(env.action_space.n))

        # Store and compute functions
        super().__init__(env=env)
        self.reset()                             # This creates a fixed grid and goal
        self._grid = self.minigrid.grid.encode() # Just to check that the grid never changes
        self._compute_model()

    def _compute_model(self):
        """Compute explicit transition and reward functions for this environment."""
        # Creating categorical distributions
        def categorical(state, probability: float):
            success_p = probability
            fail_p = (1 - probability) / len(self.states)
            return {s: success_p if s == state else fail_p for s in self.states}
        
        # Testing whether it can move there
        def can_move_there(i: int, j: int) -> bool:
            if i < 0: return False
            if j < 0: return False
            if i >= self.minigrid.width: return False
            if j >= self.minigrid.height: return False
            cell = self.minigrid.grid.get(i, j)
            if cell is not None and not cell.can_overlap(): return False
            return True

        # Compute matrices
        T: dict = defaultdict(lambda: defaultdict())
        R: dict = defaultdict(lambda: defaultdict())
        for state in self.states:
            for action in self.actions:

                # Reward
                pos = self.minigrid.grid.get(state[0], state[1])
                if pos is not None and pos.type == "goal":
                    R[state][action] = 1.0
                else:
                    R[state][action] = 0.0

                # Transition left
                if action == self.minigrid.actions.left:
                    direction = state[2]
                    direction -= 1
                    if direction < 0:
                        direction += 4
                    T[state][action] = categorical((state[0], state[1], direction), 1 - self.failure)

                # Transition right
                elif action == self.minigrid.actions.right:
                    direction = state[2]
                    direction = (direction + 1) % 4
                    T[state][action] = categorical((state[0], state[1], direction), 1 - self.failure)

                # Transition forward
                elif action == self.minigrid.actions.forward:
                    fwd_pos = np.array((state[0], state[1])) + DIR_TO_VEC[state[2]]
                    if can_move_there(*fwd_pos):
                        new_state = (fwd_pos[0], fwd_pos[1], state[2])
                    else:
                        new_state = state
                    T[state][action] = categorical(new_state, 1 - self.failure)

                else:
                    assert False, "Invalid action"

            T[state] = dict(T[state])
            R[state] = dict(R[state])
        self.T = dict(T)
        self.R = dict(R)

    def reset(self, **kwargs):
        ret = super().reset(**kwargs)
        if hasattr(self, "_grid"):
            assert (self.minigrid.grid.encode() == self._grid).all(), "The grid changed: this shouldn't happen"
        return ret
    
    def _pretty_print_T(self):
        """Prints the positive components of the transition function."""
        print("Transition function -- self.T")
        for state in self.states:
            print(f"State {state}")
            for action in self.actions:
                print(f"  action {action}")
                for state2 in self.states:
                    if self.T[state][action][state2] > 0.0:
                        print(f"    next state {state2}: {self.T[state][action][state2]}")


class DecodeObservation(gym.ObservationWrapper):
    """Decoded observation for minigrid.

    The observation is composed of agent 2D position and orientation.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.unwrapped, MiniGridEnv)
        self.minigrid = self.unwrapped
        self.observation_space = gym.spaces.MultiDiscrete(
            [self.minigrid.grid.height, self.minigrid.grid.width, 4], np.int_)

    def observation(self, observation: dict) -> np.ndarray:
        """Transform observation."""
        obs = (*self.minigrid.agent_pos, self.minigrid.agent_dir)
        return np.array(obs, dtype=np.int32)


class BinaryReward(gym.RewardWrapper):
    """1 if agent is at minigrid goal, 0 otherwise."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.unwrapped, MiniGridEnv)
        self.minigrid = self.unwrapped
        self._was_at_goal = False

    def reset(self, **kwargs):
        """Reset."""
        self._was_at_goal = False
        return super().reset(**kwargs)

    def reward(self, reward: SupportsFloat) -> float:
        """Compute reward."""
        current_cell = self.minigrid.grid.get(*self.minigrid.agent_pos)
        if current_cell is not None:
            at_goal = current_cell.type == "goal"
        else:
            at_goal = False
        rew = 1.0 if at_goal and not self._was_at_goal else 0.0
        self._was_at_goal = at_goal
        return rew


class FailProbability(gym.Wrapper):
    """Causes input actions to fail with some probability: a different action is executed."""

    def __init__(self, env: gym.Env, failure: float, seed: int, **kwargs):
        """Initialize."""
        super().__init__(env, **kwargs)
        self.failure = failure
        assert 0 <= self.failure <= 1
        self._n = int(cast(gym.spaces.Discrete, env.action_space).n)
        self._rng = random.Random(seed)

    def step(self, action):
        """Env step."""
        # Random?
        if self._rng.random() < self.failure:
            action = self._rng.randint(0, self._n - 1)
        return self.env.step(action)


def test(env: gym.Env, interactive: bool = False):
    """Environment rollouts with uniform policy for visualization.

    env: gym environment to test
    interactive: if True, the user selects the action
    """
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    def log():
        env.render()
        print("Env step")
        print("       Action:", action)
        print("  Observation:", observation)
        print("       Reward:", reward)
        print("         Done:", "terminated" if terminated else "truncated" if truncated else "False")
        print("         Info:", info)
        time.sleep(0.1)

    reward: SupportsFloat = 0.0
    action = None
    terminated = False
    truncated = False

    try:
        observation, info = env.reset()
        log()
        while True:
            # Action selection
            action = env.action_space.sample()
            if interactive:
                a = input(f"       Action (default {action}): ")
                if a:
                    action = int(a)
                if action < 0:
                    truncated = True

            # Step
            if action >= 0:
                observation, reward, terminated, truncated, info = env.step(action)
                log()

            # Reset
            if terminated or truncated:
                print("Reset")
                observation, info = env.reset()
                terminated = False
                truncated = False
                reward = 0.0
                log()
    finally:
        env.close()


if __name__ == '__main__':
    env = Empty(seed=19823283, failure=0.0, size=5, agent_start_dir=0, agent_start_pos=(1,1), render_mode='human')
    test(env, interactive=False)

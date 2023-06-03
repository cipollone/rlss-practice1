"""Environment configurations, maps and transition functions"""

import random
import time
from typing import SupportsFloat, cast

import gymnasium as gym
from minigrid import envs


class Empty(gym.Wrapper):
    """An Empty minigrid environment with explicit transition and reward functions."""

    def __init__(self, seed: int, failure=0.0, **kwargs):
        """Initialize.

        seed: random seed
        failure: failure probability of the actions (another action is executed instead).
        size: room side length
        agent_start_pos: tuple with coordinates
        agent_start_dir: north or..
        """
        env = envs.EmptyEnv(**kwargs)
        env.action_space = gym.spaces.Discrete(3)
        env = FailProbability(env, failure=failure, seed=seed)
        super().__init__(env=env)


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
                observation, info = env.reset()
                terminated = False
                truncated = False
                log()
    finally:
        env.close()


if __name__ == '__main__':
    env = Empty(seed=19823283, failure=0.5, size=5, agent_start_dir=0, agent_start_pos=(1,1), render_mode='human')
    test(env, interactive=False)

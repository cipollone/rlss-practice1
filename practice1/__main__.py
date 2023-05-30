"Main script: entry point"

import argparse
from typing import SupportsFloat
import time

import gymnasium as gym


def test_minigrid(env_name: str):
    """Testing the minigrid environment."""

    # Init
    env = gym.make(env_name, render_mode="human")
    observation, info = env.reset()
    env.render()

    reward: SupportsFloat = 0.0
    action = None
    terminated = False
    truncated = False

    def log():
        env.render()
        print("Env step")
        print("       Action:", action)
        print("  Observation:", observation)
        print("       Reward:", reward)
        print("         Done:", "terminated" if terminated else "truncated")
        print("         Info:", info)
        time.sleep(0.1)

    try:
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            log()

            if terminated or truncated:
                observation, info = env.reset()
                log()
    finally:
        env.close()


def main():
    """Main function."""

    parser = argparse.ArgumentParser("practical1")
    sub = parser.add_subparsers(dest="sub", required=True)

    # Develop
    tester = sub.add_parser("test")
    tester.add_argument("--minigrid", required=True)

    args = parser.parse_args()
    if args.sub == "test":
        test_minigrid(args.minigrid)


if __name__ == "__main__":
    main()

"Main script: entry point"

import argparse

import gymnasium as gym

from . import environments


def main():
    """Main function."""

    parser = argparse.ArgumentParser("practical1")
    sub = parser.add_subparsers(dest="sub", required=True)

    # Develop
    tester = sub.add_parser("test")
    tester.add_argument("--minigrid", required=True)
    tester.add_argument("-i", "--interactive", help="Whether to manually pass actions")

    args = parser.parse_args()

    # Do
    if args.sub == "test":
        env = gym.make(args.minigrid, render_mode="human")
        environments.test(env)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print top cumulative-time functions from a cProfile .cprof file."""
import argparse
import pstats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cprof", nargs="?", default="profile_stats.cprof")
    p.add_argument("-n", type=int, default=40)
    args = p.parse_args()
    s = pstats.Stats(args.cprof)
    s.strip_dirs().sort_stats("cumtime").print_stats(args.n)


if __name__ == "__main__":
    main()

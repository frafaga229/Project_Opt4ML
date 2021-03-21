#!/usr/bin/env python
# coding: utf-8

from mpl_toolkits.mplot3d import Axes3D
import argparse
from .visualize import toy_example


# This function to read agruments from command line
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gamma',
        help='number of epochs;',
        type=float,
        default=0.9)
    parser.add_argument(
        '--epochs',
        help='number of epochs;',
        type=int,
        default=200)
    parser.add_argument(
        '--eta',
        type=float,
        default=0.8)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Run toy example with 3 arguments epochs, gamma, eta
    toy_example(args.epochs, args.gamma, args.eta)

from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Hyperparameter optimization')
parser.add_argument('--log_dir', type=str, default="../../logs/HyperOpt/TRPO",
                    help='path to put log files')
args = parser.parse_args()

print("Printing log files to: '{}'".format(args.log_dir))
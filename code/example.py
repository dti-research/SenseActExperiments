from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Benchmarking DRL on Real World Robots')
parser.add_argument('--hid_size', type=int, default=64,
                    help='size of hidden layers (default: 64)')
parser.add_argument('--num_hid_layers', type=int, default=2,
                    help='number of hidden layers (default: 2)')
parser.add_argument('--batch_size', type=int, default=4096,
                    help='input batch size for training (default: 4096)')
parser.add_argument('--vf_stepsize', type=float, default=0.00472,
                    help='stepsize (default: 0.00472)')
parser.add_argument('--max_kl', type=float, default=0.02437,
                    help='max_kl (default: 0.02437)')
parser.add_argument('--gamma', type=float, default=0.96833,
                    help='gamma (default: 0.96833)')
parser.add_argument('--lamda', type=float, default=0.99874,
                    help='lamda (default: 0.99874)')
args = parser.parse_args()

print("{} {} {} {} {} {} {}".format(args.hid_size,
                                    args.num_hid_layers,
                                    args.batch_size,
                                    args.vf_stepsize,
                                    args.max_kl,
                                    args.gamma,
                                    args.lamda))
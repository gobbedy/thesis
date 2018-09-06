#!/usr/bin/env python3.6
import time
import logging
import portfolio_simulator
import torch
from available_cpu_count import available_cpu_count
import argparse

# note: levels are:
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG
# NOTSET

# the level is the threshold,
# eg if INFO is set, all info and "worse" (warning, error, critical") messages will be printed
# eg if ERROR is set, only ERROR and CRITICAL messages will be printed
# note: use logging cookbook for more granularity: https://docs.python.org/2/howto/logging-cookbook.html


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", type=str, help="directory in which to output any generated files", required=True)
parser.add_argument("-s", "--sanity", help="make code deterministic and use 1000-sample existing dataset for debugging", action="store_true")
parser.add_argument("-r", "--short", help="make code deterministic and use 10,000-sample existing dataset for debugging", action="store_true")
parser.add_argument("-p", "--profile", help="turn on advanced profiling", action="store_true")
parser.add_argument('-c','--compute_nodes', nargs='+', help='compute node names', required=True)
parser.add_argument('-m','--compute_nodes_pythonic', nargs='+', help='python-friendly compute node names', required=True)
args = parser.parse_args()

# configure logger
logger = logging.getLogger('portfolio_simulation')
logger.propagate = 0
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s: - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# launch simulation
logger.info(time.ctime())
logger.info("Start portfolio simulation")
num_iterations=1
#num_samples_list=[8, 16, 32, 64, 128, 256, 512, 1024, 2048]
num_samples_list=[8]

simulator = portfolio_simulator.Portfolio_simulator("simulator", args.compute_nodes, args.compute_nodes_pythonic,
                                                    num_iterations, num_samples_list, args.output_dir, args.sanity,
                                                    args.short, args.profile, "data/X_nt.npy", "data/Y_nt.npy", device)
simulator.run_simulation()
logger.info("End portfolio simulation")
logger.info(time.ctime())


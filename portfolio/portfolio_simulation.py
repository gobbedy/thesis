#!/usr/bin/env python3.6
import logging
import portfolio_simulator

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
#logging.basicConfig(level=logging.INFO)
#logging.basicConfig(filename='test.log', level=logging.INFO)


# configure logger
logger = logging.getLogger('portfolio_simulation')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s: - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# launch simulation
logger.info("Start portfolio simulation")
num_iterations=1
num_samples_list=[8]
simulator = portfolio_simulator.Portfolio_simulator("simulator", num_iterations, num_samples_list)
simulator.run_simulation()
logger.info("End portfolio simulation")

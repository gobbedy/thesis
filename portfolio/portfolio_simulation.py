#!/usr/bin/env python3.6
import portfolio_simulator


num_iterations=1
num_samples_list=[8]
simulator = portfolio_simulator.Portfolio_simulator(num_iterations, num_samples_list)
simulator.run_simulation()

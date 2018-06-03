import portfolio
import logging
from decorators import timed, profile

class Portfolio_simulator:


    def __init__(self, name, num_iterations, num_samples_list):
        self.name = name
        self.num_iterations = num_iterations
        self.num_samples_list = num_samples_list
        self.configure_logger()

    def __str__(self):
        return self.name

    def configure_logger(self):
        # create logger
        self.logger = logging.getLogger(self.name)
        self.logger.propagate = 0
        # set "verbosity"
        self.logger.setLevel(logging.INFO)
        # create console handler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        # just a placeholder -- will be updated on the fly by timed decorator
        formatter = logging.Formatter('    %(name)s - %(levelname)s: - %(message)s')
        self.ch.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(self.ch)

    def run_simulation(self):

        #epsilon=0.15
        epsilon=0.05
        lambda_=0.0
        nn_portfolio = portfolio.Nearest_neighbors_portfolio("nn_portfolio", epsilon, lambda_)
        
        # load data
        # TODO: switch back to full data
        nn_portfolio.load_data_from_csv("data/X_nt.csv.short", "data/Y_nt.csv.short")


        # NOTE: the next two lines are kept outside of the "num_iterations"
        # loop even though hyperparameter training is stochastic
        # (a random set of 80% of all samples are chosen to train and the rest to validate)
        # This is because the FI model converges to full information so we assume the
        # hyperparameters will remain constant or will negligibly change regardless of
        # which samples are training vs validation

        # If this assumption is show to be incorrect something is wrong and we need to revisit.
        
        
        # get full information hyperparameters
        nn_portfolio.compute_full_information_hyperparameters()
        
        # compute oos cost for full information model
        fi_oos_cost = nn_portfolio.compute_full_information_oos_cost()

        # outer loop especially useful at low number of samples
        for i in range(self.num_iterations):

            for num_samples in self.num_samples_list:
            
                # set number of samples for training model to train on
                nn_portfolio.set_num_samples(num_samples)
                
                # split the data into training vs validation
                nn_portfolio.split_data()

                # get training model hyperparameters
                nn_portfolio.compute_training_model_hyperparameters()

                # compute oos cost for training model
                tr_oos_cost = nn_portfolio.compute_training_model_oos_cost()
                
        print(fi_oos_cost)
        print(tr_oos_cost)

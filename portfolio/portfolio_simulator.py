import portfolio

class Portfolio_simulator:


    # TODO: check if epsilon = 0.05 is correct
    def __init__(self, num_iterations, num_samples_list):
        self.num_iterations = num_iterations
        self.num_samples_list = num_samples_list
        
   
    # TODO: move to separate class
    def run_simulation(self):

        epsilon=0.05
        lambda_=0.0
        nn_portfolio = portfolio.Nearest_neighbors_portfolio(epsilon, lambda_)
        
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
        nn_portfolio.compute_full_information_hyperparameters()
        fi_oos_cost = nn_portfolio.compute_full_information_oos_cost()

        # outer loop especially useful at low number of samples
        for i in range(self.num_iterations):

            for num_samples in self.num_samples_list:
            
                # set number of samples for training model to train on
                nn_portfolio.set_num_samples(num_samples)

                # training model hyperparameter training
                nn_portfolio.compute_training_model_hyperparameters()

                # oos sample cost calculation
                tr_oos_cost = nn_portfolio.compute_training_model_oos_cost()
                
        print(fi_oos_cost)
        print(tr_oos_cost)

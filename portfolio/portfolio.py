import random
import numpy as np
from math import sqrt, floor, ceil
import cvxpy as cp
import value_at_risk
import smoother
import hyperparameters
import logging
from decorators import timed, profile
import torch
from available_cpu_count import available_cpu_count
from time import time
import os
# import inspect
import itertools
from multiprocessing import Pool as ThreadPool
import dill as pkl
import dispy
import functools
import time

ME_DIR = os.path.dirname(os.path.realpath(__file__))

def setup_expected_responses(generated_data_filepath):

    #global random, np, sqrt, floor, ceil, cp, torch, os
    #import random
    #import numpy as np
    #from math import sqrt, floor, ceil
    #import cvxpy as cp
    #import torch
    #import os

    #import torch

    global np, os, torch
    import torch
    import numpy as np
    import os

    global x_tensor, y_tensor, k, lower_diag, xbar_tensor
    #global x, y, k, lower_diag_np, xbar_np

    compute_expected_responses_params = np.load(generated_data_filepath)

    k = compute_expected_responses_params['k']
    lower_diag = torch.from_numpy(compute_expected_responses_params['lower_diag'])
    x_tensor = torch.from_numpy(compute_expected_responses_params['x'])
    y_tensor = torch.from_numpy(compute_expected_responses_params['y'])
    xbar_tensor = torch.from_numpy(compute_expected_responses_params['xbar'])

    #lower_diag_np = compute_expected_responses_params['lower_diag']
    #x = compute_expected_responses_params['x']
    #y = compute_expected_responses_params['y']
    #xbar_np = compute_expected_responses_params['xbar']

    return 0

def compute_expected_response(j):

    #import torch

    os.environ["OMP_NUM_THREADS"] = "1"

    #lower_diag = torch.from_numpy(lower_diag_np)
    #x_tensor = torch.from_numpy(x)
    #y_tensor = torch.from_numpy(y)
    #xbar_tensor = torch.from_numpy(xbar_np)

    ## Context of interest
    xbar = xbar_tensor[j]

    ### COMPUTE SORTED NEAREST NEIGHBOR
    Xsub = (x_tensor - xbar).transpose(0, 1)

    Z = torch.trtrs(Xsub, lower_diag, upper=False)[0].transpose(0, 1)

    # mahalanobis_distances: mahalanobis distance of each X vector to Xbar
    mahalanobis_distances = torch.norm(Z, p=2, dim=1)

    ## SORT the data based on distance to xbar
    mahalanobis_distances_sorted, sorted_indices = torch.sort(mahalanobis_distances, 0)

    Y_sorted = y_tensor[sorted_indices]  # now local scope

    # adjust k to avoid eliminating equi-distant points
    inclusive_distance_boundary = mahalanobis_distances_sorted[k - 1] + 1e-7

    # cast to int because of weird incompatibility between zero-dim tensor and int in pytorch 0.4.0
    inclusive_k = int(np.searchsorted(mahalanobis_distances_sorted, inclusive_distance_boundary, side='right'))

    # get indices of nearest neighbors
    inclusive_k_nearest_neighbor_indices = np.arange(inclusive_k)

    sorted_nn_tensor = Y_sorted[inclusive_k_nearest_neighbor_indices]
    ### DONE COMPUTE SORTED NEAREST NEIGHBOR

    os.environ.pop("OMP_NUM_THREADS")

    #return torch.mean(sorted_nn_tensor, 0).size()
    return torch.mean(sorted_nn_tensor, 0)



def setup_fi_cost(x_samples_filepath, y_samples_filepath, generated_data_filepath):

    ##global random, np, sqrt, floor, ceil, cp, torch, os
    global cp, np, os, torch
    ##import random
    ##from math import sqrt, floor, ceil
    import cvxpy as cp
    import numpy as np
    import os
    import torch

    global x_tensor, y_tensor, k, lower_diag, epsilon, __lambda
    #global x_data, y_data, k, lower_diag_np, epsilon, __lambda

    x_data = np.load(x_samples_filepath)
    y_data = np.load(y_samples_filepath)
    #xbar_tensor = torch.from_numpy(x_data)
    x_tensor = torch.from_numpy(x_data)
    y_tensor = torch.from_numpy(y_data)

    fi_params = np.load(generated_data_filepath)

    k = fi_params['k']
    lower_diag = torch.from_numpy(fi_params['lower_diag'])
    #lower_diag_np = fi_params['lower_diag']
    epsilon = fi_params['epsilon']
    __lambda = fi_params['__lambda']

    return 0

def compute_optimal_portfolio(j):

    #import torch

    os.environ["OMP_NUM_THREADS"] = "1"

    #x_tensor = torch.from_numpy(x_data)
    #y_tensor = torch.from_numpy(y_data)
    #lower_diag = torch.from_numpy(lower_diag_np)


    # 1. get nearest neighbors
    xbar = x_tensor[j]

    #sorted_nn_tensor = compute_sorted_nearest_neighbors(global_arrays, xbar)


    ### COMPUTE SORTED NEAREST NEIGHBOR
    Xsub = (x_tensor - xbar).transpose(0, 1)


    Z = torch.trtrs(Xsub, lower_diag, upper=False)[0].transpose(0, 1)

    # mahalanobis_distances: mahalanobis distance of each X vector to Xbar
    mahalanobis_distances = torch.norm(Z, p=2, dim=1)

    ## SORT the data based on distance to xbar
    mahalanobis_distances_sorted, sorted_indices = torch.sort(mahalanobis_distances, 0)

    Y_sorted = y_tensor[sorted_indices]  # now local scope

    # adjust k to avoid eliminating equi-distant points
    inclusive_distance_boundary = mahalanobis_distances_sorted[k - 1] + 1e-7

    # cast to int because of weird incompatibility between zero-dim tensor and int in pytorch 0.4.0
    inclusive_k = int(np.searchsorted(mahalanobis_distances_sorted, inclusive_distance_boundary, side='right'))

    # get indices of nearest neighbors
    inclusive_k_nearest_neighbor_indices = np.arange(inclusive_k)

    sorted_nn_tensor = Y_sorted[inclusive_k_nearest_neighbor_indices]
    ### DONE COMPUTE SORTED NEAREST NEIGHBOR

    nearest_neighbors = sorted_nn_tensor.numpy()

    # 2. set up optimization problem
    loss_len = len(nearest_neighbors)
    num_assets = y_tensor.size(1)
    z = cp.Variable(num_assets)
    L = cp.Variable(loss_len)
    b = cp.Variable(1)

    # note: just "sum" instead of long sum_entries command appears to work

    obj = cp.Minimize(cp.sum(L)/loss_len)

    # Constraints
    # long only and unit leverage
    constrs = [z>=0, sum(z)==1]

    for i in range(loss_len):
        # this must define the loss function L. Second part obvious
        # not sure why first part is the same?
        constrs = constrs + [L[i] >= (1-1/epsilon)*b - (__lambda+1/epsilon)*sum(cp.multiply(nearest_neighbors[i], z))]
        constrs = constrs + [L[i] >= b - __lambda*sum(cp.multiply(nearest_neighbors[i, :], z))]

    # find optimal z, VaR (b) -- which minimizes total cost
    # this minimum total cost (over hitorical points) is problem.optval
    problem=cp.Problem(obj, constrs)


    # 3. now, optimize
    problem.solve(solver=cp.ECOS)

    os.environ.pop("OMP_NUM_THREADS")

    #print (problem.value, z.value, b.value, problem.status)
    #'''
    #return (0,1,2)
    return (problem.value, z.value, b.value, problem.status)

class Nearest_neighbors_portfolio:

    def __init__(self, name, compute_nodes, compute_nodes_pythonic, epsilon, __lambda, output_dir, x_samples_filename, y_samples_filename, sanity=False, short=False, profile=False):
        self.name = name
        self.compute_nodes = compute_nodes
        self.compute_nodes_pythonic = compute_nodes_pythonic
        self.epsilon = epsilon
        self.__lambda = __lambda
        self.output_dir = output_dir
        self.x_samples_filename = x_samples_filename
        self.y_samples_filename = y_samples_filename
        self.sanity = sanity
        self.short = short
        self.profile = profile

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


    def set_num_samples(self, num_samples):

        self.num_samples = num_samples


    @timed
    def split_data(self):

        if self.sanity or self.short:

            # train on first num_samples_in_dataset samples; note: each row is a sample!
            self.X_tr = self.X_data[0:self.num_samples]
            self.Y_tr = self.Y_data[0:self.num_samples]

            # all the non-training data is considered "validation" data -- but this is actually out of sample data! almost all the data is out of sample
            self.X_val = self.X_data[self.num_samples:]
            self.Y_val = self.Y_data[self.num_samples:]

        else:
        
            # Training data
            train_perm = sorted(random.sample(range(len(self.X_data)), self.num_samples))
            self.X_tr = self.X_data[train_perm]
            self.Y_tr = self.Y_data[train_perm]

            # Validation data
            val_perm = sorted(list(set(range(len(self.X_data))) - set(train_perm)))
            self.X_val = self.X_data[val_perm]
            self.Y_val = self.Y_data[val_perm]

    @timed
    def compute_full_information_hyperparameters(self):

        self.hyperparameters_fi = self.compute_hyperparameters(self.Y_data, self.X_data)


    # can find analytically?
    @timed
    def compute_full_information_oos_cost(self):

        # see compute_expected_responses for explanation of this hack
        #global compute_full_information_oos_cost_globals
        #compute_full_information_oos_cost_globals = type('test', (), {})()
        #compute_full_information_oos_cost_globals = Compute_full_information_oos_cost_vars(torch.from_numpy(self.X_data), torch.from_numpy(self.X_data), torch.from_numpy(self.Y_data), self.hyperparameters_fi, self.epsilon, self.__lambda)
        #compute_full_information_oos_cost_globals.__module__ = '__main__'
        #compute_full_information_oos_cost_globals.Xbar_tensor = torch.from_numpy(self.X_data)
        #compute_full_information_oos_cost_globals.X_tensor = torch.from_numpy(self.X_data)
        #compute_full_information_oos_cost_globals.Y_tensor = torch.from_numpy(self.Y_data)
        #compute_full_information_oos_cost_globals.hyperparameters_object = self.hyperparameters_fi
        #compute_full_information_oos_cost_globals.epsilon = self.epsilon
        #compute_full_information_oos_cost_globals.__lambda = self.__lambda
        generated_data_filepath = self.output_dir + "/autogen/full_information_params.npz"
        np.savez(generated_data_filepath, k=self.hyperparameters_fi.k, lower_diag=self.hyperparameters_fi.upper_diag.transpose(0, 1),
                 epsilon=self.epsilon, __lambda=self.__lambda)

        #num_cores = int(available_cpu_count())
        #print(num_cores)
        #os.environ["OMP_NUM_THREADS"] = "1"

        '''
        pool = ThreadPool(num_cores)
        args_iterable = zip(itertools.repeat("compute_full_information_oos_cost_globals"), range(len(self.X_data)))
        optimal_portfolio_list = pool.starmap(Nearest_neighbors_portfolio.compute_optimal_portfolio,args_iterable,10)
        '''

        '''
        from pathos.pp import ParallelPythonPool as Pool
        ppservers = ("nia0859.scinet.local:1234",)
        pool = Pool(num_cores*2, servers=ppservers)


        #optimal_portfolio_list = pool.starmap(Nearest_neighbors_portfolio.compute_optimal_portfolio,
        #                zip(itertools.repeat("compute_full_information_oos_cost_globals"), range(len(self.X_data))),
        #                                      1000)

        #optimal_portfolio_list = pool.starmap(Nearest_neighbors_portfolio.compute_optimal_portfolio,
        #                zip(itertools.repeat("compute_full_information_oos_cost_globals"), range(len(self.X_data))))


        #args_iterable = zip(itertools.repeat("compute_full_information_oos_cost_globals"), range(len(self.X_data)))
        #optimal_portfolio_list = pool.starmap(Nearest_neighbors_portfolio.compute_optimal_portfolio,args_iterable)


        args_iterable = zip(itertools.repeat("compute_full_information_oos_cost_globals"), range(len(self.X_data)))
        optimal_portfolio_results_object = pool.amap(Nearest_neighbors_portfolio.compute_optimal_portfolio, *map(list, zip(*args_iterable)) )
        optimal_portfolio_list = optimal_portfolio_results_object.get()
        '''
        #pkl_filename = 'compute_full_information_oos_cost_globals.pkl'
        #with open(pkl_filename, 'wb') as handle:
        #    pkl.dump(compute_full_information_oos_cost_globals, handle)

        x_samples_filepath = ME_DIR + '/' + self.x_samples_filename
        y_samples_filepath = ME_DIR + '/' + self.y_samples_filename

        # change working directory temporarily to force JobCluster command to dump in the proper output directory
        original_working_dir = os.getcwd()
        os.chdir(self.output_dir + '/' + 'dispy')

        # tell dispy where all the compute nodes are and set them up using setup command
        cluster = dispy.JobCluster(compute_optimal_portfolio, nodes=self.compute_nodes_pythonic, setup=functools.partial(setup_fi_cost, x_samples_filepath, y_samples_filepath, generated_data_filepath))
        #cluster = dispy.JobCluster(compute_optimal_portfolio, nodes=["nia1189.scinet.local", ], setup=setup)

        # return to original working dir to avoid any unintended effects from dir change
        os.chdir(original_working_dir)

        jobs = []
        for i in range(len(self.X_data)):
        #for i in range(10):
            self.i = i
            job = cluster.submit(i) # it is sent to a node for executing 'compute'
            job.id = i # store this object for later use
            jobs.append(job)

        full_information_oos_costs = np.empty(len(self.X_data))
        for idx, job in enumerate(jobs):
            job() # wait for job to finish
            full_information_oos_costs[idx] = job.result[0]

        cluster.close()

        # relaunch dispynodes because of this bug: https://github.com/pgiri/dispy/issues/143
        cmd_str = 'launch_remote_dispynodes.sh ' + self.output_dir + ' ' + ' '.join(self.compute_nodes)
        os.system(cmd_str)
        time.sleep(3)


        '''
        pool.close()
        pool.join()
        '''
        #os.environ.pop("OMP_NUM_THREADS")


        # extract only costs from list of portfolio problem tuples (cost is first element of every tuple
        # see https://stackoverflow.com/a/31297256/8112889
        #full_information_oos_costs_list = list(zip(*optimal_portfolio_list))[0]

        # convert costs list to torch tensor
        #full_information_oos_costs = torch.stack(full_information_oos_costs_list)
        #full_information_oos_costs = np.array(full_information_oos_costs_list)

        #return torch.mean(full_information_oos_costs)
        return np.mean(full_information_oos_costs)
        

    @timed
    def compute_training_model_hyperparameters(self):

        # Find an appropriate nn smoother using training data:
        #     -- learn the distance function itself
        #         -- from which we get weighter based on heuristically chosen bandwidth and smoother)
        #     -- learn the number of nearest neighbours
        logging.info("Getting hyperparameters for training NN model...")
        self.hyperparameters_tr = self.compute_hyperparameters(self.Y_tr, self.X_tr)


    @timed
    def compute_training_model_oos_cost(self):

        # see compute_expected_responses for explanation of this hack
        global compute_training_model_oos_cost_globals
        compute_training_model_oos_cost_globals = type('', (), {})()
        compute_training_model_oos_cost_globals.Xbar_tensor = torch.from_numpy(self.X_val)
        compute_training_model_oos_cost_globals.X_tensor = torch.from_numpy(self.X_tr)
        compute_training_model_oos_cost_globals.Y_tensor = torch.from_numpy(self.Y_tr)
        compute_training_model_oos_cost_globals.hyperparameters_object = self.hyperparameters_tr
        compute_training_model_oos_cost_globals.epsilon = self.epsilon
        compute_training_model_oos_cost_globals.__lambda = self.__lambda

        num_cores = int(available_cpu_count())
        os.environ["OMP_NUM_THREADS"] = "1"
        pool = ThreadPool(num_cores)
        optimal_portfolio_list = pool.starmap(Nearest_neighbors_portfolio.compute_optimal_portfolio,
                        zip(itertools.repeat("compute_training_model_oos_cost_globals"), range(len(self.X_val))))


        # extract only costs from list of portfolio problem tuples (cost is first element of every tuple
        # see https://stackoverflow.com/a/31297256/8112889
        #####training_oos_costs_list = list(zip(*optimal_portfolio_list))[0]

        # convert costs list to torch tensor
        #full_information_oos_costs = torch.stack(full_information_oos_costs_list)
        #####training_oos_costs = np.array(training_oos_costs_list)

        pool.close()
        pool.join()
        os.environ.pop("OMP_NUM_THREADS")

        tr_learner_oos_cost_true=0
        for idx, optimal_portfolio in enumerate(optimal_portfolio_list):

            c_tr, z_tr, b_tr, s_tr = optimal_portfolio
            x_val = self.X_val[idx]

            # find b (VaR) analytically
            b=value_at_risk.value_at_risk(x_val, z_tr, self.epsilon)

            # find true Y|X (returns Y distribution with weights)
            training_loss_fnc = lambda y: self.loss(z_tr, b, y)
            training_loss = np.apply_along_axis(training_loss_fnc, 1, self.Y_data)
            c_tr_true = self.compute_expected_responses(training_loss, self.X_data, x_val, self.hyperparameters_fi)

            tr_learner_oos_cost_true += c_tr_true

        return tr_learner_oos_cost_true/len(self.X_val)


    @timed
    def load_data(self):

        #self.X_data = np.loadtxt(x_csv_filename, delimiter=",")
        #self.Y_data = np.loadtxt(y_csv_filename, delimiter=",")
        self.X_data = np.load(self.x_samples_filename)
        self.Y_data = np.load(self.y_samples_filename)



#    @timed
    def loss(self, z, b, y):
    
        return b + 1/self.epsilon*max(-np.dot(z, y)-b, 0)-self.__lambda*np.dot(z, y)

    '''
#    @timed
    def mahalanobis(self, x1, x2, A):
        \'''
        sqrt( (x1-x2)inv(A)(x1-x2) )
        \'''

        # Note: can get performance gain setting check_finite to false
        # Note2: what is returned is the lower left matrix despite lower=false, why?
        (A, lower) = cho_factor(A, overwrite_a=True, check_finite=True)

        # Distance function -- note that distance function -- smoother built on top
        return np.sqrt((x1-x2) @ cho_solve((A, lower), x1-x2, overwrite_b=True, check_finite=True))
    '''

    @timed
    def compute_hyperparameters(self, Y, X, p=0.2, smoother_list=[smoother.Smoother("Naive")]):

        # num rows X -- ie num samples
        num_samples_in_dataset = np.size(X, 0)

        # num cols X -- ie num covariates
        num_covariates = np.size(X, 1)

        # num cols of Y -- ie num assets
        num_assets = np.size(Y, 1)

        logging.debug("## Problem Parameters")
        logging.debug("1. Number of samples num_samples_in_dataset = " + str(num_samples_in_dataset))
        logging.debug("2. Label dimension : " + str(num_assets))
        logging.debug("3. Covariate dimension : " + str(num_assets))
        logging.debug("## Hyperparameter optimization")
        logging.debug("1. Proportion VALIDATION/TOTAL data =" + str(p))
        logging.debug("2. Considered Smoothers : " + str(smoother_list))

        # Compute covariance of covariates
        # TODO: check the math, why identity -- is this really mahalanobis?
        epsilonX = np.cov(X.T, bias=True) + np.identity(num_covariates)/num_samples_in_dataset

        upper_diag = torch.from_numpy(epsilonX)
        torch.potrf(upper_diag, out=upper_diag)

        # hyperparameters

        # TODO: add this unused julia code for NW portfolio?
        #D = [d(X[i, :], mean_X) for i in range(0,num_samples_in_dataset)]

        k_list = np.unique(np.round(np.linspace(max(1, floor(sqrt(num_samples_in_dataset)/1.5)), min(ceil(sqrt(num_samples_in_dataset)*1.5), num_samples_in_dataset), 20).astype('int')))

        # pick 20% of the original (training) samples as your validation set -- note: sorting not necessary
        if self.sanity or self.short:
            val = range(round(num_samples_in_dataset*p))
        else:
            val = sorted(random.sample(range(num_samples_in_dataset), round(num_samples_in_dataset*p)))

        # the remaining 80% is your new "training" set
        train = sorted(list(set(range(num_samples_in_dataset)) - set(val)))

        logging.debug("Number of k to test: " + str(len(k_list)))

        shortest_distance = -1
        for test_smoother in smoother_list:
            for test_k in k_list:

                # TODO: add this unused julia code for NW portfolio?
                #bandwidth_list = logspace(log10(minimum(D)), log10(maximum(D)), 10)
                if test_smoother == "Naive":
                    bandwidth_list = [1]

                for bandwidth in bandwidth_list:

                    test_hyperparameters = hyperparameters.Hyperparameters(test_k, test_smoother, upper_diag, bandwidth)

                    logging.debug("Smoother function : " + str(test_hyperparameters))
                    logging.debug("Number of neighbors : k = " + str(test_k))

                    
                    # find E[Y|xbar] for all X in validation set
                    expected_responses = self.compute_expected_responses(Y[train], X[train], X[val],
                                                                         test_hyperparameters)

                    # sum distance of all these E[Y|xbar] to true Y (respectively)
                    model_distance = np.sum((Y[val]-expected_responses)**2)

                    # the shortest such distance corresponds to most accurate model, ie 
                    # this model has best hyperparameters, so we store them
                    if model_distance < shortest_distance or shortest_distance == -1:
                        shortest_distance = model_distance
                        shortest_distance_hyperparameters = test_hyperparameters

        return shortest_distance_hyperparameters


    @timed
    @profile
    def compute_expected_responses(self, Y, X, Xbar, hyperparameters_object):

        """
        Arguments:

            Y: historical 'response' variable (typically asset returns)
            X: historical covariates
            Xbar: observation ("today's" covariate) -- can be interpreted as X context of interest
            hyperparameters_object: number of nearest neighbors k, smoother function, upper_diagonal, bandwidth
              -- uuper diagonal is Cholesky factor of mahalanobis matrix

        Returns:

            expected_response: expected response given Xbar observation/context


        Description:

            1. Find the k nearest neighbors of Xbar inside X, using mahalanobis distance
            2. Adjust k so that points just outside k-set which have equal distance as kth point are included
               -- call this adjusted k, "inclusive_k"
            3. Assign weights based on mahalanobis distance, a bandwidth, and a smoothing function
               -- Points further from xbar generally have smaller weights (naive smoother has equal weights)
               -- Smoother transforms distance to weights (eg for gaussian smoother, zero distance is center
                  of gaussian curve and further distances fall with distance from center)
               -- Higher bandwidth reduces distance as seen by smoothing function
                  -- depending on smoother this could have different effects (for example, for squaref uniform,
                     will give zero weight to fewer points)

        """
        num_samples_in_dataset = np.size(Y, 0)

        if Y.ndim == 2:
          num_assets = np.size(Y, 1)
        else: # Y.ndim ==1
          num_assets = 1

        num_covariates = np.size(X,1)

        if Xbar.ndim == 2:
            num_observations = np.size(Xbar, 0)
        else: # Xbar.dim == 1
            num_observations = 1

        # creating a global class is a hack workaround until pathos.multiprocessing supports
        # passing local large arrays: https://github.com/uqfoundation/pathos/issues/145
        # TODO: update if/when pathos adds local array support
        # This will require passing these arrays as arguments to starmap (will need to check if that works)
        # or making compute_expected_response nested (this is know to work but very slow until pathos fixes)
        # probably better still would be to make it a closure if that works: https://www.learnpython.org/en/Closures
        #   -- it will be defined elsewhere, with only the arrays being passed in and the function is generated with
        # only the j argument. -- will need to check if the resulting function truly creates shared arrays
        # and doesn't clone

        #global compute_expected_responses_globals
        #compute_expected_responses_globals = type('', (), {})() # see https://stackoverflow.com/a/19476841/8112889
        #compute_expected_responses_globals.Xbar_tensor = torch.from_numpy(Xbar).view(num_observations, -1)
        #compute_expected_responses_globals.X_tensor = torch.from_numpy(X)
        #compute_expected_responses_globals.Y_tensor = torch.from_numpy(Y)
        #compute_expected_responses_globals.hyperparameters_object = hyperparameters_object

        #'''
        generated_data_filepath = self.output_dir + '/autogen/compute_expected_responses_params.npz'
        np.savez(generated_data_filepath, k=hyperparameters_object.k, lower_diag=hyperparameters_object.upper_diag.transpose(0, 1),
                 x=X, y=Y, xbar=Xbar.reshape(num_observations,-1))

        # change working directory temporarily to force JobCluster command to dump in the proper output directory
        original_working_dir = os.getcwd()
        os.chdir(self.output_dir + '/' + 'dispy')

        # tell dispy where all the compute nodes are and set them up using setup command
        cluster = dispy.JobCluster(compute_expected_response, nodes=self.compute_nodes_pythonic, setup=functools.partial(setup_expected_responses, generated_data_filepath))
        #cluster = dispy.JobCluster(compute_optimal_portfolio, nodes=["nia1189.scinet.local", ], setup=setup)

        # return to original working dir to avoid any unintended effects from dir change
        os.chdir(original_working_dir)

        jobs = []

        for i in range(num_observations):
        #for i in range(10):
            self.i = i
            job = cluster.submit(i) # it is sent to a node for executing 'compute'
            job.id = i # store this object for later use
            jobs.append(job)

        #expected_responses_list = np.empty(len(self.Xbar))
        expected_responses_list = torch.empty(num_observations, num_assets)
        for idx, job in enumerate(jobs):
            job() # wait for job to finish
            #BLA")
            #print(job.result)
            #print("BLU")
            #exit()
            expected_responses_list[idx] = job.result

        cluster.close()
        
        # relaunch dispynodes because of this bug: https://github.com/pgiri/dispy/issues/143
        cmd_str = 'launch_remote_dispynodes.sh ' + self.output_dir + ' ' + ' '.join(self.compute_nodes)
        os.system(cmd_str)
        time.sleep(3)
        #'''

        '''
        ts = time()
        num_cores = int(available_cpu_count())
        #num_cores = 32
        #print(num_cores)
        #exit()
        os.environ["OMP_NUM_THREADS"] = "1"
        pool = ThreadPool(num_cores)
        #pool.map(Nearest_neighbors_portfolio.compute_expected_response, range(num_observations))
        expected_responses_list = pool.starmap(Nearest_neighbors_portfolio.compute_expected_response,
                                zip(itertools.repeat("compute_expected_responses_globals"), range(num_observations)))

        pool.close()
        pool.join()
        te = time()
        #os.environ.pop("OMP_NUM_THREADS")

        expected_responses = torch.stack(expected_responses_list).numpy()
        '''

        expected_responses = expected_responses_list.numpy()

        return expected_responses

    @staticmethod
    def compute_expected_response(global_arrays_class_name, j):

        ## Context of interest
        #####xbar = compute_expected_responses_globals.Xbar_tensor[j]
        global_arrays = globals()[global_arrays_class_name]
        xbar = global_arrays.Xbar_tensor[j]

        sorted_nearest_neighbors = Nearest_neighbors_portfolio.compute_sorted_nearest_neighbors(global_arrays, xbar)

        return torch.mean(sorted_nearest_neighbors, 0)

    @staticmethod
    def compute_sorted_nearest_neighbors(global_arrays, xbar):

        # x1 - x2
        #####Xsub = (compute_expected_responses_globals.X_tensor - xbar).transpose(0, 1)
        #x_tensor = globals()[global_arrays_class_name].X_tensor
        Xsub = (global_arrays.X_tensor - xbar).transpose(0, 1)

        #####lower_diag = compute_expected_responses_globals.hyperparameters_object.upper_diag.clone().transpose(0, 1)
        hyperparameters_obj = global_arrays.hyperparameters_object
        lower_diag = hyperparameters_obj.upper_diag.clone().transpose(0, 1)
        Z = torch.trtrs(Xsub, lower_diag, upper=False)[0].transpose(0, 1)

        # mahalanobis_distances: mahalanobis distance of each X vector to Xbar
        # L2 norm -- note: square root not necessary, since we only car about sorting not absolute actual number
        # but since speed of this call is not a bottleneck, this is fine
        mahalanobis_distances = torch.norm(Z, p=2, dim=1)

        ## SORT the data based on distance to xbar
        mahalanobis_distances_sorted, sorted_indices = torch.sort(mahalanobis_distances, 0)

        #####Y_sorted = compute_expected_responses_globals.Y_tensor[sorted_indices]  # now local scope
        Y_sorted = global_arrays.Y_tensor[sorted_indices]  # now local scope

        # adjust k to avoid eliminating equi-distant points
        #####k = compute_expected_responses_globals.hyperparameters_object.k
        k = global_arrays.hyperparameters_object.k
        inclusive_distance_boundary = mahalanobis_distances_sorted[k - 1] + 1e-7

        # cast to int because of weird incompatibility between zero-dim tensor and int in pytorch 0.4.0
        inclusive_k = int(np.searchsorted(mahalanobis_distances_sorted, inclusive_distance_boundary, side='right'))

        # get indices of nearest neighbors
        inclusive_k_nearest_neighbor_indices = np.arange(inclusive_k)

        '''
        # This is a template for applying non-naive smoother to weigh nearest-neighbor points
        # The code was functionally tested and can be used as is except the for loop for which a form of 
        # broadcasting should be found, if possible for the smoother in question (for speed)           
        weights = mahalanobis_distances[inclusive_k_nearest_neighbor_indices] / hyperparameters_object.bandwidth
        for i in inclusive_k_nearest_neighbor_indices:
            weights[i] = hyperparameters_object.smoother(weights[i])
        weights = weights / sum(weights)

        # unsqueeze(1)/view(inclusive_k, 1) for broadcast multiplication to work as expected;
        # double() needed because smoother tested (naive) spits out a float (1.0) value instead of double.
        # double() likely won't be needed if/when this actually needs to be used
        # since smoother will likely divide/multiply an existing double() and therefore return a double
        weights = weights.view(inclusive_k, 1).double()

        # E[Y|xbar], ie weighted/"smoothed" average of the Y[i,:] corresponding to the nearest inclusive_k X
        expected_response[j] = torch.sum(weights * Y_tensor[inclusive_k_nearest_neighbor_indices].view(
            inclusive_k, num_assets), 0)

        '''

        return Y_sorted[inclusive_k_nearest_neighbor_indices]


#    @timed
    @staticmethod
    def compute_optimal_portfolio(global_arrays_class_name, j):
        import random
        import numpy as np
        from math import sqrt, floor, ceil
        import cvxpy as cp
        import smoother
        import hyperparameters
        import torch
        import os

        os.environ["OMP_NUM_THREADS"] = "1"
        #if j%100 == 0:
        #    print("compute_optimal_portfolio: start k-nearest: ", j)

        # 1. get nearest neighbors
        global_arrays = globals()[global_arrays_class_name]
        xbar = global_arrays.Xbar_tensor[j]

        sorted_nn_tensor = Nearest_neighbors_portfolio.compute_sorted_nearest_neighbors(global_arrays, xbar)
        nearest_neighbors = sorted_nn_tensor.numpy()

        # 2. set up optimization problem
        loss_len = len(nearest_neighbors)
        num_assets = global_arrays.Y_tensor.size(1)
        z = cp.Variable(num_assets)
        L = cp.Variable(loss_len)
        b = cp.Variable(1)

        # note: just "sum" instead of long sum_entries command appears to work

        #print("compute_optimal_portfolio: start A: ", j)
        obj = cp.Minimize(cp.sum(L)/loss_len)

        # Constraints
        # long only and unit leverage
        constrs = [z>=0, sum(z)==1]

        epsilon = global_arrays.epsilon
        __lambda = global_arrays.__lambda

        for i in range(loss_len):
            # this must define the loss function L. Second part obvious
            # not sure why first part is the same?
            constrs = constrs + [L[i] >= (1-1/epsilon)*b - (__lambda+1/epsilon)*sum(cp.multiply(nearest_neighbors[i], z))]
            constrs = constrs + [L[i] >= b - __lambda*sum(cp.multiply(nearest_neighbors[i, :], z))]

        # find optimal z, VaR (b) -- which minimizes total cost
        # this minimum total cost (over hitorical points) is problem.optval

        #print("compute_optimal_portfolio: start B: ", j)
        problem=cp.Problem(obj, constrs)


        # 3. now, optimize

        # note: ECOS solver would probably be picked by cvxpy
        # TODO: run with default, see if it picks a faster one / compare speed of different solvers
        # note: more solvers can be added to core cvxpy
        # see "choosing a solver": http://www.cvxpy.org/tutorial/advanced/index.html
        # note that SCS can use GPUs -- See https://github.com/cvxgrp/cvxpy/issues/245
        # can Boyd's POGS solver be used?
        # look into warm start -- make sure it is leveraged
        #print("compute_optimal_portfolio: start optimization: ", j)
        problem.solve(solver=cp.ECOS)

        os.environ.pop("OMP_NUM_THREADS")

        return (problem.value, z.value, b.value, problem.status)

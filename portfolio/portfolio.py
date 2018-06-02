import random
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from math import sqrt, floor, ceil
import cvxpy as cp
import value_at_risk
import smoother
import weighter
import logging
import sys
from decorators import timed, profile
from numba import jit

class Nearest_neighbors_portfolio:


    def __init__(self, name, epsilon, __lambda):
        self.name = name
        self.epsilon = epsilon
        self.__lambda = __lambda
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
        formatter = logging.Formatter('    %(name)s - %(levelname)s: - %(message)s')
        self.ch.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(self.ch)


    def set_num_samples(self, num_samples):

        self.num_samples = num_samples


    @timed
    def split_data(self):

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

        self.d_fi, self.k_fi, self.Sn_fi = self.compute_hyperparameters(self.Y_data, self.X_data)


    # can find analytically?
    @timed
    def compute_full_information_oos_cost(self):

        fi_learner_oos_cost = 0
        for x_val in self.X_data:
            c_fi, z_fi, b_fi, s_fi = self.optimize_nearest_neighbors_portfolio(self.Y_data, self.X_data, self.d_fi, self.k_fi, self.Sn_fi, x_val)
            fi_learner_oos_cost += c_fi


        return fi_learner_oos_cost/len(self.X_data)
        

    @timed
    def compute_training_model_hyperparameters(self):

        # Find an appropriate nn smoother using training data (learn the distance function itself, the number of nearest neighbours, and the type of smoother)
        logging.info("Getting hyperparameters for training NN model...")
        self.d_tr, self.k_tr, self.Sn_tr = self.compute_hyperparameters(self.Y_tr, self.X_tr)


    @timed
    def compute_training_model_oos_cost(self):

        tr_learner_oos_cost_true=0
        logging.info("Performing out of sample test for both (full information and training) NN models with " \
                     + str(len(self.X_val)) + " oos samples...")
        for idx, x_val in enumerate(self.X_val):

            if idx%10 == 0:
                logging.debug("out: " + str(idx))

            # oos cost of training learner (based on its knowledge of historical data) -- estimate
            c_tr, z_tr, b_tr, s_tr = self.optimize_nearest_neighbors_portfolio(self.Y_tr, self.X_tr, self.d_tr, self.k_tr, self.Sn_tr, x_val)

            # find b (VaR) analytically
            b=value_at_risk.value_at_risk(x_val, z_tr, self.epsilon)

            # find true Y|X (returns Y distribution with weights)
            training_loss_fnc = lambda y: self.loss(z_tr, b, y)
            training_loss = np.apply_along_axis(training_loss_fnc, 1, self.Y_data)
            c_tr_true = self.compute_expected_response(training_loss, self.X_data, x_val, self.d_fi, self.k_fi, self.Sn_fi)

            tr_learner_oos_cost_true += c_tr_true

        return tr_learner_oos_cost_true/len(self.X_val)


    @timed
    def load_data_from_csv(self, x_csv_filename, y_csv_filename):

        self.X_data = np.loadtxt(x_csv_filename, delimiter=",")
        self.Y_data = np.loadtxt(y_csv_filename, delimiter=",")


    @timed
    def loss(self, z, b, y):
    
        return b + 1/self.epsilon*max(-np.dot(z, y)-b, 0)-self.__lambda*np.dot(z, y)

#    @timed
    def mahalanobis(self, x1, x2, A):
        '''
        sqrt( (x1-x2)inv(A)(x1-x2) )
        '''

        # Note: can get performance gain setting check_finite to false (what is returned is the lower left matrix despite lower=false, why?)
        (A, lower) = cho_factor(A, overwrite_a=True, check_finite=True)

        # Distance function -- note that distance function -- smoother built on top
        return np.sqrt((x1-x2) @ cho_solve((A, lower), x1-x2, overwrite_b=True, check_finite=True))


    @timed
    def compute_hyperparameters(self, Y, X, p=0.2, smoother_list=[smoother.Smoother("Naive")]):

        # num rows X -- ie num samples
        n = np.size(X, 0)

        # num cols X -- ie num covariates
        d_x = np.size(X, 1)

        # num cols of Y -- ie num assets
        d_y = np.size(Y, 1)

        logging.debug("## Problem Parameters")
        logging.debug("1. Number of samples n = " + str(n))
        logging.debug("2. Label dimension : " + str(d_y))
        logging.debug("3. Covariate dimension : " + str(d_y))
        logging.debug("## Hyperparameter optimization")
        logging.debug("1. Proportion VALIDATION/TOTAL data =" + str(p))
        logging.debug("2. Considered Smoothers : " + str(smoother_list))

        # Compute covariance of covariates
        # TODO: check the math, why identity -- is this really mahalanobis?
        epsilonX = np.cov(X.T, bias=True) + np.identity(d_x)/n

        # Note: can get performance gain setting check_finite to false (what is returned is the lower left matrix despite lower=false, why?)
        (epsilonX, lower) = cho_factor(epsilonX, overwrite_a=True, check_finite=True)

        # Distance function -- note that distance function -- smoother built on top
        d = lambda x1, x2: np.sqrt((x1-x2) @ cho_solve((epsilonX, lower), x1-x2, overwrite_b=True, check_finite=True))

        # distance function
        #d = lambda x1, x2: self.mahalanobis(x1, x2, epsilonX)

        # hyperparameters

        # TODO: add this unused julia code for NW portfolio?
        #D = [d(X[i, :], mean_X) for i in range(0,n)]

        k_all = np.unique(np.round(np.linspace(max(1, floor(sqrt(n)/1.5)), min(ceil(sqrt(n)*1.5), n), 20).astype('int')))

        # pick 20% of the original (training) samples as your validation set -- note: sorting not necessary
        val = sorted(random.sample(range(n), round(n*p)))

        # the remaining 80% is your new "training" set
        train = sorted(list(set(range(n)) - set(val)))

        logging.debug("Number of k to test: " + str(len(k_all)))

        val_errors_star = (-1, -1, -1)
        for __smoother in smoother_list:
            for k in k_all:

                # TODO: add this unused julia code for NW portfolio?
                #bandwidth_list = logspace(log10(minimum(D)), log10(maximum(D)), 10)
                if __smoother == "Naive":
                    bandwidth_list = [1]

                for bandwidth in bandwidth_list:

                    __weighter = weighter.Weighter(__smoother, d, bandwidth)

                    logging.debug("Smoother function : " + str(__weighter))
                    logging.debug("Number of neighbors : k = " + str(k))

                    # find E[Y|X] ie best guess for Y given your that your learner is trained on this new smaller "training" set
                    Yp = self.compute_expected_response(Y[train], X[train], X[val], d, k, __weighter)

                    # compare the learner's best guess for Y to Y|X of the validation set
                    # ie find total distance between learner's Y|X and true Y|X -- store corresponding smoother and number of k
                    total_distance = np.sum((Y[val]-Yp)**2)
                    if total_distance < val_errors_star[0] or val_errors_star[0] == -1:
                        val_errors_star = (total_distance, __weighter, k)

        # notice that learning distance function only uses data
        # learning smoother (weigher) if not preset, and learning number of k-nearest points, use training data AND Y|X learner

        return (d, val_errors_star[2], val_errors_star[1])


    """
        compute_expected_response(Y, X, Xbar, d, k, __weighter )

    # Arguments

        1. `Y` : Array of observed responses
        2. `X` : Covariate data
        3. `xbar` : Context of interest
        4. `d` : Distance metric
        5. `k` : Number of considered neighbors
        6. `__weighter` : Weighter function

    # Returns
        1. `Yp` : Nearest neighbors prediction

    """
    # find E[Y|X]:
    # 1. Find k nearest neighbors based on mahalanobis distance
    # 2. Adjust k in case points just outside k-set have equal distance as kth point
    # 3. Assign weights based on mahalanobis distance and a bandwidth
    # 4. Apply smoother on top of these weights
    @timed
    @profile
    def compute_expected_response(self, Y, X, Xbar, d, k, __weighter):

        n = np.size(Y, 0)

        if Y.ndim == 2:
          d_y = np.size(Y, 1)
        else: # Y.ndim ==1
          d_y = 1

        if Xbar.ndim == 2:
            m = np.size(Xbar, 0)
        else: # Xbar.dim == 1
            m = 1

        logging.debug("# Hyper Parameters")
        logging.debug("1. Number of nearest neighbors k = " + str(k))
        logging.debug("2. Smoother function S = " + str(__weighter))
        logging.debug("# Problem Parameters")
        logging.debug("1. Number of samples n = " + str(n))
        logging.debug("2. Label dimension : " + str(d_y))
        logging.debug("3. Number of contexts : " + str(m))

        Yp = np.zeros((m, d_y))
        for j in range(m):

            ## Context of interest
            # if I only input one observation, m=1 and xbar = Xbar
            if Xbar.ndim == 2:
                xbar = Xbar[j]
            else: # Xbar.dim == 1
                xbar = Xbar
                
            
            dist = self.numba_test(d, X, xbar)

            '''
            ## SORT the data based on distance to xbar
            # TODO: apply_along_axis is NOT fast -- use numba (perhaps with original loop) for speedup
            dist_from_xbar = lambda x1: d(x1, xbar)


            
            # I AM HERE: try making this a separate function with a loop and jitting that function (yes, that simple -- who knows)
            dist = np.apply_along_axis(dist_from_xbar, 1, X)
            # dist = [d(X[i], xbar ) for i in range(n)]
            '''

            perm = np.argsort(dist)
            dist = dist[perm]
            Y = Y[perm] # now local scope
            X = X[perm] # now local scope

            ## Define set of points of interest
            R_star = dist[k-1]+1e-7

            # adjusted k to avoid eliminating equi-distant points
            # Nk is an array of indices
            Nk = np.where(dist <= R_star)[0]

            # note: __weighter is a Weighter object

            # TODO: apply_along_axis is NOT fast -- use numba (perhaps with original loop) for speedup
            weight_from_xbar = lambda x1: __weighter(x1, xbar)
            S = np.apply_along_axis(weight_from_xbar, 1, X[Nk])
            #S = [weight_fcn(X[i], xbar) for i in Nk]

            ## Prediction --> this is E[Y|X]!!!
            # Take the weighted average of all the Y[i,:] -- where i in nearest adjusted_k

            #Y_nearestk_weighted = (S*Y[Nk].T).T
            #Yp[j] = mean(Y_nearestk_weighted, 0)

            # weighted average of all the Y[i,:] -- where i in nearest adjusted_k
            Yp[j] = np.mean((S*Y[Nk].T).T, 0)

        return Yp

    @jit
    def numba_test(self, d, X, xbar):
        dist=np.empty(len(X))
        for i in range(len(X)):
            dist[i] = d(X[i], xbar)  
        return dist
    
    @timed
    def optimize_nearest_neighbors_portfolio(self, Y, X, d, k, __weighter, xbar):

        n = np.size(Y, 0)

        d_y = np.size(Y, 1)

        logging.debug("## Problem Parameters")
        logging.debug("1. Risk level CVaR epsilon = " + str(self.epsilon))
        logging.debug("2. Risk / Reward trade off __lambda = " + str(self.__lambda))
        logging.debug("## Problem dimensions ")
        logging.debug("1. Number of samples n = " + str(n))
        logging.debug("2. Label dimension : " + str(d_y))
        logging.debug("## Hyper parameters")
        logging.debug("1. Number of nearest neighbors k = " + str(k))
        logging.debug("2. Weighter function __weighter = " + str(__weighter))

        ## SORT the data based on distance to xbar        
        # TODO: apply_along_axis is NOT fast -- use numba (perhaps with original loop) for speedup
        dist_from_xbar = lambda x1: d(x1, xbar)
        dist = np.apply_along_axis(dist_from_xbar, 1, X)
        #dist = [d(X[i], xbar ) for i in range(n)]

        perm = np.argsort(dist)
        dist = dist[perm]
        Y_nn = Y[perm]
        X_nn = X[perm]

        # rather than do k nearest neighbors, also include any points
        # that are basically equal to the kth point (or within 1e-7)
        # this avoids making the threshold arbitrarily cut off equal points
        R_star = dist[k-1]+1e-7
        Nk = np.where(dist <= R_star)[0]

        # OPTIMIZATION FORMULATION
        z = cp.Variable(d_y)
        L = cp.Variable(len(Nk))
        b = cp.Variable(1)

        # Objective -- L is loss L(y,z) -- heavier weight to points closer to xbar
        # ie with greater distance ie higher __weighter
        # since this is a minimization not clear why need to divide by the sum of all distances?

        # TODO: apply_along_axis is NOT fast -- use numba (perhaps with original loop) for speedup
        weight_from_xbar = lambda x1: __weighter(x1, xbar)

        S = np.apply_along_axis(weight_from_xbar, 1, X_nn[Nk])

        # note: just "sum" instead of long sum_entries command appears to work
        obj = cp.Minimize(sum(cp.multiply(S,L))/sum(S))

        # Constraints
        # long only and unit leverage
        constrs = [z>=0, sum(z)==1]


        for i in Nk:
            # this must define the loss function L. Second part obvious
            # not sure why first part is the same?
            constrs = constrs + [L[i] >= (1-1/self.epsilon)*b - (self.__lambda+1/self.epsilon)*sum(cp.multiply(Y_nn[i], z))]
            constrs = constrs + [L[i] >= b - self.__lambda*sum(cp.multiply(Y_nn[i, :], z))]

        # find optimal z, VaR (b) -- which minimizes total cost
        # this minimum total cost (over hitorical points) is problem.optval
        problem=cp.Problem(obj, constrs)

        # note: ECOS solver would probably be picked by cvxpy
        # TODO: run with default, see if it picks a faster one / compare speed of different solvers
        # note: more solvers can be added to core cvxpy
        # see "choosing a solver": http://www.cvxpy.org/tutorial/advanced/index.html 
        # note that SCS can use GPUs! See https://github.com/cvxgrp/cvxpy/issues/245
        # can Boyd's POGS solver be used?
        # look into warm start -- make sure it is leveraged!
        problem.solve(solver=cp.ECOS)

        return (problem.value, z.value, b.value, problem.status)

import random
import numpy as np
from math import sqrt, floor, ceil
import cvxpy as cp
import value_at_risk
import smoother
import weighter
import logging
from decorators import timed, profile
import torch
import platform

from multiprocessing.dummy import Pool as ThreadPool 


class Nearest_neighbors_portfolio:


    def __init__(self, name, epsilon, __lambda, sanity=False, profile=False, device=torch.device('cpu')):
        self.name = name
        self.epsilon = epsilon
        self.__lambda = __lambda
        self.configure_logger()
        self.sanity=sanity
        self.profile=profile
        self.device=device

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

        if self.sanity:

            # train on first n samples; note: each row is a sample!
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

        self.k_fi, self.weighter_fi = self.compute_hyperparameters(self.Y_data, self.X_data)


    # can find analytically?
    @timed
    def compute_full_information_oos_cost(self):

        fi_learner_oos_cost = 0
        for x_val in self.X_data:
            c_fi, z_fi, b_fi, s_fi = self.optimize_nearest_neighbors_portfolio(self.Y_data, self.X_data,
                                                                               self.k_fi, self.weighter_fi, x_val)
            fi_learner_oos_cost += c_fi


        return fi_learner_oos_cost/len(self.X_data)
        

    @timed
    def compute_training_model_hyperparameters(self):

        # Find an appropriate nn smoother using training data:
        #     -- learn the distance function itself
        #         -- from which we get weighter based on heuristically chosen bandwidth and smoother)
        #     -- learn the number of nearest neighbours
        logging.info("Getting hyperparameters for training NN model...")
        self.k_tr, self.weighter_tr = self.compute_hyperparameters(self.Y_tr, self.X_tr)


    @timed
    def compute_training_model_oos_cost(self):

        tr_learner_oos_cost_true=0
        logging.info("Performing out of sample test for both (full information and training) NN models with " \
                     + str(len(self.X_val)) + " oos samples...")
        for idx, x_val in enumerate(self.X_val):

            if idx%10 == 0:
                logging.debug("out: " + str(idx))

            # oos cost of training learner (based on its knowledge of historical data) -- estimate
            c_tr, z_tr, b_tr, s_tr = self.optimize_nearest_neighbors_portfolio(self.Y_tr, self.X_tr,
                                                                               self.k_tr, self.weighter_tr, x_val)

            # find b (VaR) analytically
            b=value_at_risk.value_at_risk(x_val, z_tr, self.epsilon)

            # find true Y|X (returns Y distribution with weights)
            training_loss_fnc = lambda y: self.loss(z_tr, b, y)
            training_loss = np.apply_along_axis(training_loss_fnc, 1, self.Y_data)
            c_tr_true = self.compute_expected_response(training_loss, self.X_data, x_val, self.k_fi,
                                                       self.weighter_fi)

            tr_learner_oos_cost_true += c_tr_true

        return tr_learner_oos_cost_true/len(self.X_val)


    @timed
    def load_data_from_csv(self, x_csv_filename, y_csv_filename):

        self.X_data = np.loadtxt(x_csv_filename, delimiter=",")
        self.Y_data = np.loadtxt(y_csv_filename, delimiter=",")


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

        # Note: can get performance gain setting check_finite to false
        # Note2: what is returned is the lower left matrix despite lower=false, why?
        ##(epsilonX, lower) = cho_factor(epsilonX, overwrite_a=True, check_finite=True)
        upper_diag = torch.from_numpy(epsilonX)
        torch.potrf(upper_diag, out=upper_diag)

        # Distance function -- note that distance function -- smoother built on top
        
        ##d = lambda x1, x2: np.sqrt((x1-x2) @ cho_solve((epsilonX, lower), x1-x2, overwrite_b=True, check_finite=True))
        ###d = lambda x1, x2: torch.sqrt(torch.mm((x1-x2).transpose(0,1), torch.potrs((x1-x2), upper_diag) ))
        ###def d(x1, x2):
        ###    torch.sqrt(torch.mm((x1-x2).view(1,len(x1)), torch.potrs((x1-x2).view(len(x1),1), upper_diag) ))

        # distance function
        ##d = lambda x1, x2: self.mahalanobis(x1, x2, epsilonX)

        # hyperparameters

        # TODO: add this unused julia code for NW portfolio?
        #D = [d(X[i, :], mean_X) for i in range(0,n)]

        k_list = np.unique(np.round(np.linspace(max(1, floor(sqrt(n)/1.5)), min(ceil(sqrt(n)*1.5), n), 20).astype('int')))

        # pick 20% of the original (training) samples as your validation set -- note: sorting not necessary
        if self.sanity:
            val = range(round(n*p))
        else:
            val = sorted(random.sample(range(n), round(n*p)))

        # the remaining 80% is your new "training" set
        train = sorted(list(set(range(n)) - set(val)))

        logging.debug("Number of k to test: " + str(len(k_list)))

        shortest_distance = -1
        for __smoother in smoother_list:

            # TODO: add this unused julia code for NW portfolio?
            #bandwidth_list = logspace(log10(minimum(D)), log10(maximum(D)), 10)
            if __smoother == "Naive":
                bandwidth_list = [1]

            for bandwidth in bandwidth_list:

                hyperparameter_object = weighter.Weighter(__smoother, bandwidth)

                compute_expected_response_current_training_set = lambda x1: self.compute_expected_response(Y[train], X[train], X[val], x1, hyperparameter_object)

                pool = ThreadPool(16)
                Yp_dist = np.array(pool.map(compute_expected_response_current_training_set, k_list))
                type(Yp_dist)
                exit()
                pool.close()
                pool.join()

                for k in k_list:

                    logging.debug("Smoother function : " + str(hyperparameter_object))
                    logging.debug("Number of neighbors : k = " + str(k))

                    # find E[Y|xbar] for all X in validation set
                    Yp = self.compute_expected_response(Y[train], X[train], X[val], k, hyperparameter_object)

                    # sum distance of all these E[Y|xbar] to true Y (respectively)
                    model_distance = np.sum((Y[val]-Yp)**2)

                    # the shortest such distance corresponds to most accurate model, ie
                    # this model has best weighter/k combination, so we store weighter/k
                    if model_distance < shortest_distance or shortest_distance == -1:
                        shortest_distance = model_distance
                        hyperparameters = (k, hyperparameter_object)

        return hyperparameters


    """
        compute_expected_response(Y, X, Xbar, k, hyperparameter_object )

    # Arguments

        1. `Y` : Array of observed responses
        2. `X` : Covariate data
        3. `xbar` : Context of interest
        4. `d` : Distance metric
        5. `k` : Number of considered neighbors
        6. `hyperparameter_object` : Hyperparameter object

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
    def compute_expected_response(self, Y, X, Xbar, k, hyperparameter_object):

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
        logging.debug("2. Smoother function S = " + str(hyperparameter_object))
        logging.debug("# Problem Parameters")
        logging.debug("1. Number of samples n = " + str(n))
        logging.debug("2. Label dimension : " + str(d_y))
        logging.debug("3. Number of contexts : " + str(m))

        Xbar_tensor=torch.from_numpy(Xbar)
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)

        Yp = np.zeros((m, d_y))

        if platform.system() == "UNUSED":
        #if not platform.system() == "Windows":
            # note: this does not use "expanded" k
            index = faiss.IndexFlatL2(X_tensor.size(1))
            index.add(X)
            distances, nearest_neighbor_index_matrix = index.search(Xbar, k)
            for idx, nearest_neighbor_index_array in enumerate(nearest_neighbor_index_matrix):
                S = np.empty_like(nearest_neighbor_index_array)
                for jdx, nearest_neighbor_index in enumerate(nearest_neighbor_index_array):
                    S[jdx] = weight_from_xbar(distances[idx, jdx])
                Yp[idx] = np.mean((S * Y_tensor[nearest_neighbor_index_array].numpy().T).T, 0)

        else:

            for j in range(m):

                ## Context of interest
                # if I only input one observation, m=1 and xbar = Xbar
                if Xbar.ndim == 2:
                    xbar = Xbar_tensor[j]
                else: # Xbar.dim == 1
                    xbar = Xbar_tensor


                ## SORT the data based on distance to xbar
                # TODO: apply_along_axis is NOT fast -- use pytorch (perhaps with original loop) for speedup
                ###dist_from_xbar = lambda x1: d(x1.view(len(x1),1), xbar.view(len(xbar),1))



                ##pool = ThreadPool(16)
                ##dist = np.array(pool.map(dist_from_xbar, X_tensor))
                ##pool.close()
                ##pool.join()

                # x1 - x2
                Xsub = X_tensor - xbar
                Z=torch.empty(X_tensor.size(), device=self.device)
                for i in range(n):
                    # z = Linv * (x - xbar), where L is lower diagonal matrix
                    Z[i]=torch.trtrs(Xsub[i], hyperparameter_object.upper_diag.transpose(0,1), upper=False)[0].transpose(0,1)


                # L2 norm -- note: square root not necessary, algorithm that doesn't take it could be faster
                # but since speed is not the objective here, this is fine
                #dist = torch.norm(X_tensor)
                dist = torch.norm(Z, p=2, dim=1)


                ###dist = np.empty(n)
                ###X_tensor=torch.from_numpy(X)
                ###for i in range(n):
                ###    dist[i] = dist_from_xbar(X_tensor[i])

                ##dist = np.apply_along_axis(dist_from_xbar, 1, X)
                # dist = [d(X[i], xbar ) for i in range(n)]

                dist, perm = torch.sort(dist, 0)
                Y_tensor = Y_tensor[perm] # now local scope
                X_tensor = X_tensor[perm] # now local scope


                ###perm = np.argsort(dist)
                ###dist = dist[perm]
                ###Y = Y[perm] # now local scope
                ###X = X[perm] # now local scope

                # Define set of points of interest
                R_star = dist[k-1]+1e-7

                # adjusted k to avoid eliminating equi-distant points
                # Nk is an array of indices
                ###Nk = np.where(dist <= R_star)[0]
                Nk = torch.nonzero(dist <= R_star).cpu().squeeze().numpy()


                # TODO: apply_along_axis is NOT fast -- use pytorch (perhaps with original loop) for speedup
                weight_from_xbar = lambda x1: hyperparameter_object.smoother(x1 / hyperparameter_object.bandwidth)
                #self.smoother(self.distance(x1, x2) / self.bandwidth)
                if Nk.ndim > 0:
                    S = np.empty_like(Nk)
                    for i in Nk:
                        S[i] = weight_from_xbar(dist[i])
                # handle case where Nk is scalar ("0-d array" technically)
                else:
                    S = weight_from_xbar(dist[Nk])
                ##S = np.apply_along_axis(weight_from_xbar, 1, X[Nk])
                #S = [weight_fcn(X[i], xbar) for i in Nk]

                ## Prediction --> this is E[Y|X]!!!
                # Take the weighted average of all the Y[i,:] -- where i in nearest adjusted_k

                #Y_nearestk_weighted = (S*Y[Nk].T).T
                #Yp[j] = mean(Y_nearestk_weighted, 0)

                # weighted average of all the Y[i,:] -- where i in nearest adjusted_k
                #print(Y_tensor.shape)
                #print(Nk.shape)
                #print(Y_tensor[Nk].shape)
                #print(Y_tensor[Nk].numpy().shape)
                Yp[j] = np.mean((S * Y_tensor[Nk].numpy().T).T, 0)
                #Yp[j] = np.mean((S * Y_tensor[Nk].transpose(0,1)).transpose(0,1), 0)
                #Yp[j] = np.mean((S * Y[Nk].T).T, 0)


        return Yp
    
#    @timed
    def optimize_nearest_neighbors_portfolio(self, Y, X, k, hyperparameter_object, xbar):

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
        logging.debug("2. Hyperparameter object hyperparameter_object = " + str(hyperparameter_object))

        ## SORT the data based on distance to xbar        

        ## Context of interest
        # if I only input one observation, m=1 and xbar = Xbar
        xbar_tensor = torch.from_numpy(xbar)

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        ##pool = ThreadPool(16)
        ##dist = np.array(pool.map(dist_from_xbar, X_tensor))
        ##pool.close()
        ##pool.join()

        # x1 - x2
        Xsub = X_tensor - xbar_tensor
        Z = torch.empty(X_tensor.size())
        for i in range(n):
            # z = Linv * (x - xbar), where L is lower diagonal matrix
            Z[i] = torch.trtrs(Xsub[i], hyperparameter_object.upper_diag.transpose(0, 1), upper=False)[0].transpose(0, 1)

        # L2 norm -- note: square root not necessary, algorithm that doesn't take it could be faster
        # but since speed is not the objective here, this is fine
        # dist = torch.norm(X_tensor)
        dist = torch.norm(Z, p=2, dim=1)

        dist, perm = torch.sort(dist, 0)
        Y_nn = Y_tensor[perm].numpy() # now local scope
        X_nn = X_tensor[perm].numpy()  # now local scope

        # Define set of points of interest
        R_star = dist[k - 1] + 1e-7

        # adjusted k to avoid eliminating equi-distant points
        # Nk is an array of indices
        ###Nk = np.where(dist <= R_star)[0]
        Nk = torch.nonzero(dist <= R_star).squeeze().numpy()

        # TODO: apply_along_axis is NOT fast -- use pytorch (perhaps with original loop) for speedup
        weight_from_xbar = lambda x1: hyperparameter_object.smoother(x1 / hyperparameter_object.bandwidth)
        # self.smoother(self.distance(x1, x2) / self.bandwidth)
        if Nk.ndim > 0:
            S = np.empty_like(Nk)
            for i in Nk:
                S[i] = weight_from_xbar(dist[i])
        # handle case where Nk is scalar ("0-d array" technically)
        else:
            S = weight_from_xbar(dist[Nk])

        # Objective -- L is loss L(y,z) -- heavier weight to points closer to xbar
        # ie with greater distance ie higher
        # since this is a minimization not clear why need to divide by the sum of all distances?

        # OPTIMIZATION FORMULATION
        z = cp.Variable(d_y)
        L = cp.Variable(len(Nk))
        b = cp.Variable(1)

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
        # note that SCS can use GPUs -- See https://github.com/cvxgrp/cvxpy/issues/245
        # can Boyd's POGS solver be used?
        # look into warm start -- make sure it is leveraged
        problem.solve(solver=cp.ECOS)

        return (problem.value, z.value, b.value, problem.status)

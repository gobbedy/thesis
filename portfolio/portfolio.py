import random
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from math import sqrt, floor, ceil
import cvxpy as cp
import value_at_risk
import smoother
import weighter

class Nearest_neighbors_portfolio:


    # TODO: check if epsilon = 0.05 is correct
    def __init__(self, num_samples, epsilon, lambda_):
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.lambda_ = lambda_
        
   
    # TODO: move to separate class
    def run_simulation(self):
        # load data
        self.load_data_from_csv("X_nt.csv.short", "Y_nt.csv.short")
        
        # hyperparameter training
        self.compute_full_information_hyperparameters()
        self.compute_training_model_hyperparameters()
        
        # oos sample cost calculation
        self.compute_full_information_oos_cost()
        self.compute_training_model_oos_cost()
        

    def compute_full_information_hyperparameters(self):

        self.d_fi, self.k_fi, self.Sn_fi = self.nearest_neighbors_v(self.Y_data, self.X_data, verbose=False)
        

    def compute_training_model_hyperparameters(self):

        # Training data
        ####perm = shuffle(1:size(self.X_data, 1))

        # shuffle data
        ####self.X_data = self.X_data[perm, :]
        ####self.Y_data = self.Y_data[perm, :]


        # PUT THIS BACK
        #train_perm = sorted(random.sample(range(len(X)), self.num_samples))
        #self.X_tr = self.X_data[train_perm]
        #self.Y_tr = self.Y_data[train_perm]

        # PUT THIS BACK
        #val_perm = sorted(list(set(range(len(X))) - set(train_perm)))
        #self.X_val = self.X_data[val_perm]
        #self.Y_val = self.Y_data[val_perm]


        # train on first n samples; note: each row is a sample!
        self.X_tr = self.X_data[0:self.num_samples]
        self.Y_tr = self.Y_data[0:self.num_samples]

        # all the non-training data is considered "validation" data -- but this is actually out of sample data! almost all the data is out of sample
        self.X_val = self.X_data[self.num_samples:]
        self.Y_val = self.Y_data[self.num_samples:]

        # Find an appropriate nn smoother using training data (learn the distance function itself, the number of nearest neighbours, and the type of smoother)
        print("Getting hyperparameters for training NN model...")
        self.d_tr, self.k_tr, self.Sn_tr = self.nearest_neighbors_v(self.Y_tr, self.X_tr, verbose=False)


    # can find analytically?
    def compute_full_information_oos_cost(self):

        fi_learner_oos_cost = 0
        # PUT THIS BACK for x_val in self.X_data:
        for x_val in self.X_data[8:58]:
            c_fi, z_fi, b_fi, s_fi = self.portfolio_nn_nominal(self.Y_data, self.X_data, self.d_fi, self.k_fi, self.Sn_fi, x_val, verbose=False)
            fi_learner_oos_cost += c_fi


        # PUT THIS BACK fi_learner_oos_cost = fi_learner_oos_cost/len(X_data)
        fi_learner_oos_cost = fi_learner_oos_cost/50
        
        print(fi_learner_oos_cost)


    def load_data_from_csv(self, x_csv_filename, y_csv_filename):

        # PUT THIS BACK X_data = np.loadtxt(filename, delimiter=",")
        # PUT THIS BACK Y_data = np.loadtxt(filename, delimiter=",")
        self.X_data = np.loadtxt(x_csv_filename, delimiter=",", skiprows=1)
        self.Y_data = np.loadtxt(y_csv_filename, delimiter=",", skiprows=1)


    def loss(self, z, b, y):
    
        return b + 1/self.epsilon*max(-np.dot(z, y)-b, 0)-self.lambda_*np.dot(z, y)


    def nearest_neighbors_v(self, Y, X, p=0.2, S_all=[smoother.Smoother("Naive")], verbose=False):

        # num rows X -- ie num samples
        n = np.size(X, 0)

        # num cols X -- ie num covariates
        d_x = np.size(X, 1)

        # num cols of Y -- ie num assets
        d_y = np.size(Y, 1)

        if verbose:
            print("# Start Nearest Neighbors Hyper Parameter Validation")
            print("****************************************************")
            print("## Problem Parameters")
            print("1. Number of samples n = ", n)
            print("2. Label dimension : ", d_y)
            print("3. Covariate dimension : ", d_y)
            print("## Hyperparameter optimization")
            print("1. Proportion VALIDATION/TOTAL data =", p)
            print("2. Considered Smoothers : ", S_all)

        # Compute covariance of covariates

        # average of each column of X -- ie average value of each covariate
        mean_X = np.mean(X, 0)
        epsilonX = np.cov(X.T, bias=True) + np.identity(d_x)/n

        #epsilonX = factorize(epsilonX)
        #epsilonX = np.linalg.cholesky(epsilonX)
        # Note: can get performance gain setting check_finite to false (what is returned is the lower left matrix despite lower=false, why?)
        (epsilonX, lower) = cho_factor(epsilonX, overwrite_a=True, check_finite=True)

        # Distance function -- note that distance function -- smoother built on top
        d = lambda x1, x2: np.sqrt((x1-x2) @ cho_solve((epsilonX, lower), x1-x2, overwrite_b=True, check_finite=True))

        # hyperparameters

        # TODO: add this unused julia code in for NW portfolio?
        #D = [d(X[i, :], mean_X) for i in range(0,n)]

        k_all = np.unique(np.round(np.linspace(max(1, floor(sqrt(n)/1.5)), min(ceil(sqrt(n)*1.5), n), 20).astype('int')))

        # pick 20% of the original (training) samples as your validation set -- note: sorting not necessary
        ## PUT THIS BACK AFTER TESTING val = sorted(random.sample(range(n), round(n*p)))
        val = range(round(n*p))
        #print(val)
        #val = sample(1:n, round(Int, n*p), replace=false)

        # the remaining 80% is your new "training" set
        train = sorted(list(set(range(n)) - set(val)))

        if verbose:
             print("Number of k to test: ", np.size(k_all,1))

        val_errors_star = (-1, -1, -1)
        # S is Smoother: see smoother.py
        for S in S_all:
            for k in k_all:

                #h_all = logspace(log10(minimum(D)), log10(maximum(D)), 10)
                if S == "Naive":
                    h_all = [1]

                for hn in h_all:

                    # Sn is "weighter": just a tuple containing these three variables
                    Sn = weighter.Weighter(S, d, hn)

                    if verbose:
                        print("*****VALIDATION*****")
                        print("Smoother function : ", Sn)
                        print("Number of neighbors : k = ", k)

                    # find E[Y|X] ie best guess for Y given your that your learner is trained on this new smaller "training" set
                    Yp = self.nearest_neighbors_learner(Y[train], X[train], X[val], d, k, Sn, verbose=False)

                    # compare the learner's best guess for Y to Y|X of the validation set
                    # ie find total distance between learner's Y|X and true Y|X -- store corresponding smoother and number of k
                    total_distance = np.sum((Y[val]-Yp)**2)
                    if total_distance < val_errors_star[0] or val_errors_star[0] == -1:
                        val_errors_star = (total_distance, Sn, k)

        # notice that learning distance function only uses data
        # learning smoother (weigher) if not preset, and learning number of k-nearest points, use training data AND Y|X learner

        if verbose:
            print("********************************************************")
            print("*** End Nearest Neighbors Hyper Parameter Validation ***")

        #print(val_errors_star)
        #print(d, val_errors_star[2], val_errors_star[1])
        return (d, val_errors_star[2], val_errors_star[1])


    """
        nearest_neighbors_learner(Y, X, Xbar, d, k, Sn[, verbose] )

    # Arguments

        1. `Y` : Array of observed responses
        2. `X` : Covariate data
        3. `xbar` : Context of interest
        4. `d` : Distance metric
        5. `k` : Number of considered neighbors
        6. `Sn` : Weighter function
        7. `verbose` : Verbosity

    # Returns
        1. `Yp` : Nearest neighbors prediction

    """
    def nearest_neighbors_learner(self, Y, X, Xbar, d, k, Sn, verbose=False):

        n = np.size(Y, 0)

        if Y.ndim == 2:
          d_y = np.size(Y, 1)
        else: # Y.ndim ==1
          d_y = 1

        if Xbar.ndim == 2:
            m = np.size(Xbar, 0)
        else: # Xbar.dim == 1
            m = 1

        if verbose:
            print("# Start Nearest Neighbors Learning")
            print("**********************************")
            print("# Hyper Parameters")
            print("1. Number of nearest neighbors k = ", k)
            print("2. Smoother function S = ", Sn)
            print("# Problem Parameters")
            print("1. Number of samples n = ", n)
            print("2. Label dimension : ", d_y)
            print("3. Number of contexts : ", m)

        Yp = np.zeros((m, d_y))
        for j in range(m):

            ## Context of interest
            # if I only input one observation, m=1 and xbar = Xbar
            if Xbar.ndim == 2:
                xbar = Xbar[j]
            else: # Xbar.dim == 1
                xbar = Xbar

            ## SORT the data based on distance to xbar
            # TODO: apply_along_axis is NOT fast -- use numba (perhaps with original loop) for speedup
            dist_from_xbar = lambda x1: d(x1, xbar)
            dist = np.apply_along_axis(dist_from_xbar, 1, X)
            #print(dist)
            #print(d(X[6], xbar))
            # dist = [d(X[i], xbar ) for i in range(n)]

            perm = np.argsort(dist)
            dist = dist[perm]
            Y = Y[perm] # now local scope
            X = X[perm] # now local scope

            ## Define set of points of interest
            R_star = dist[k-1]+1e-7

            # adjusted k to avoid eliminating equi-distant points
            # Nk is an array of indices
            Nk = np.where(dist <= R_star)[0]

            # note: Sn is a Weighter object

            # TODO: apply_along_axis is NOT fast -- use numba (perhaps with original loop) for speedup
            weight_from_xbar = lambda x1: Sn(x1, xbar)
            S = np.apply_along_axis(weight_from_xbar, 1, X[Nk])
            #print(S)
            #S = [weight_fcn(X[i], xbar) for i in Nk]

            ## Prediction --> this is E[Y|X]!!!
            # Take the weighted average of all the Y[i,:] -- where i in nearest adjusted_k

            #Y_nearestk_weighted = (S*Y[Nk].T).T
            #Yp[j] = mean(Y_nearestk_weighted, 0)

            Yp[j] = np.mean((S*Y[Nk].T).T, 0)


        if verbose:
            print("**************************************")
            print("*** End Nearest Neighbors Learning ***")

        #print(Yp)
        return Yp
    
    
    def portfolio_nn_nominal(self, Y, X, d, k, Sn, xbar, verbose=False):

        #print("PORTFOLIO NN PYTHON START")

        n = np.size(Y, 0)

        d_y = np.size(Y, 1)
        #if Y.ndim == 2:
        #  d_y = np.size(Y, 1)
        #else: # Y.ndim ==1
        #  d_y = 1

        if verbose:
            print("# Start Portfolio Nominal NN Formulation")
            print("******************************************")
            print("## Problem Parameters")
            print("1. Risk level CVaR epsilon = ", self.epsilon)
            print("2. Risk / Reward trade off lambda_ = ", self.lambda_)
            print("## Problem dimensions ")
            print("1. Number of samples n = ", n)
            print("2. Label dimension : ", d_y)
            print("## Hyper parameters")
            print("1. Number of nearest neighbors k = ", k)
            print("2. Weighter function Sn = ", Sn)

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
        # ie with greater distance ie higher Sn
        # since this is a minimization not clear why need to divide by the sum of all distances?

        # TODO: apply_along_axis is NOT fast -- use numba (perhaps with original loop) for speedup
        weight_from_xbar = lambda x1: Sn(x1, xbar)

        S = np.apply_along_axis(weight_from_xbar, 1, X_nn[Nk])

        # note: just "sum" instead of long sum_entries command appears to work
        obj = cp.Minimize(sum(cp.multiply(S,L))/sum(S))

        # Constraints
        # long only and unit leverage
        constrs = [z>=0, sum(z)==1]


        for i in Nk:
            # this must define the loss function L. Second part obvious
            # not sure why first part is the same?
            constrs = constrs + [L[i] >= (1-1/self.epsilon)*b - (self.lambda_+1/self.epsilon)*sum(cp.multiply(Y_nn[i], z))]
            constrs = constrs + [L[i] >= b - self.lambda_*sum(cp.multiply(Y_nn[i, :], z))]

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

        if verbose:
            print("*** End Portfolio Nominal NN Formulation ***")
            print("**********************************************")

        #print("python portfolio nn nominal:")
        #print(problem.value)
        #print(z.value)
        #print(problem.status)
        #print("end python portfolio nn nominal")

        #print("PORTFOLIO NN PYTHON END")

        return (problem.value, z.value, b.value, problem.status)




    def compute_training_model_oos_cost(self):

        print("Testing training model (TM) vs full information model (FIM) for TM trained with ", self.num_samples)
        print(" training samples...")

        tr_learner_oos_cost_estimate=0
        tr_learner_oos_cost_true=0
        #PUT THIS BACK num_oos_samples = size(self.Y_val, 1)
        # OR BETTER in the loop just do for x_val in self.X_val:
        num_oos_samples = 50
        print("Performing out of sample test for both (full information and training) NN models with ", num_oos_samples, end='')
        print(" oos samples...")
        for i in range(num_oos_samples):

            if i%10 == 0:
                print("out: ", i)

            # oos cost of training learner (based on its knowledge of historical data) -- estimate
            c_tr, z_tr, b_tr, s_tr = self.portfolio_nn_nominal(self.Y_tr, self.X_tr, self.d_tr, self.k_tr, self.Sn_tr, self.X_val[i], verbose=False)

            # find b (VaR) analytically
            b=value_at_risk.value_at_risk(self.X_val[i], z_tr, self.epsilon)

            # find true Y|X (returns Y distribution with weights)
            training_loss_fnc = lambda y: self.loss(z_tr, b, y)
            training_loss = np.apply_along_axis(training_loss_fnc, 1, self.Y_data)
            c_tr_true = self.nearest_neighbors_learner(training_loss, self.X_data, self.X_val[i], self.d_fi, self.k_fi, self.Sn_fi)

            tr_learner_oos_cost_true += c_tr_true

        tr_learner_oos_cost_true = tr_learner_oos_cost_true/num_oos_samples
        print(tr_learner_oos_cost_true)

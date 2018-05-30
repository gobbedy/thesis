#!/usr/bin/python3.4

import numpy as np

dx=3
dy=12

# A. Set sigma_u

# 1. create sigma_u and set diagonal
sigma_u = 8/7*np.eye(dx,dx)

# 2. set odd/even elements offsets

# grab sigma_u size
#sigma_u_sz = np.ma.size(sigma_u)

# set even/odd offsets
sigma_u.flat[0::2] += -1/7
sigma_u.flat[1::2] += 1/7

# 3. "minimize" sigma_u
# multiply whole matrix by 0.05
sigma_u *= 0.05

# B. Set phi1, phi2
phi1 = np.array([[0.5,-0.9,0],[1.1,-0.7,0],[0,0,0.5]])
phi2 = np.array([[0,-0.5,0],[-0.5,0,0],[0,0,0]])

# C. Set theta1, theta2
theta1 = np.array([[0.4,0.8,0],[-1.1,-0.3,0],[0,0,0]])
theta2 = np.array([[0,-0.8,0],[-1.1,0,0],[0,0,0]])

# D. Set A, B
A = 0.025 * np.array([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]])
B = 0.075 * np.array([[0,-1,-1],[-1,0,-1],[-1,-1,0],[0,-1,1],[-1,0,1],[-1,1,0],[0,1,-1],[1,0,-1],[1,-1,0],[0,1,1],[1,0,1],[1,1,0]])

# Get 100,000 samples of the U process (each row is a sample)
num_samples=100000
U = np.random.multivariate_normal(np.zeros(dx), sigma_u, num_samples)

X = np.zeros([num_samples, dx])
X[0] = U[0]
# note: np.matmul(ph1, X[0] = npmatmu(X[0],np.transpose(phi1))
X[1] = np.matmul(phi1, X[0]) + U[1] + np.matmul(theta1, U[0])

for i in range(2, num_samples):
  X[i] = U[i] + np.matmul(theta1, U[i-1]) + np.matmul(theta2, U[i-2]) + np.matmul(phi1, X[i-1]) + np.matmul(phi2, X[i-2])

delta = np.random.standard_normal((num_samples, dy))
epsilon =  np.random.standard_normal((num_samples, dy))

Y = np.zeros([num_samples, dy])
for i in range(0, num_samples):
  # note X[i][:,none] is the weird numpy way of doing transpose of X[i] (because 1d-array)
  # probably there's a more efficient ways to do this using broadcasting
  Y[i] = np.matmul(X[i], np.transpose(A)) + np.multiply(np.sum(A,axis=1), delta[i]/4) + np.multiply(np.matmul(X[i], np.transpose(B)), epsilon[i])

# checked with
#for j in range(100000):
#    for i in range(12):
#        a=X[j] + delta[j,i]/4
#        b=np.matmul(A[i],a[:,None])
#        c=np.matmul(B[i],X[j][:,None])
#        d=c*epsilon[j,i]
#np.testing.assert_array_almost_equal(Y, Y_check)

std_dev_Y = np.std(Y)
mean_Y = np.mean(Y)
print(std_dev_Y)
print(mean_Y)

Z=(Y-mean_Y)*(1/std_dev_Y)

std_dev_Z = np.std(Z)
mean_Z = np.mean(Z)
print(std_dev_Z)
print(mean_Z)

Z=(1/std_dev_Y)*Y - mean_Y

std_dev_Z = np.std(Z)
mean_Z = np.mean(Z)
print(std_dev_Z)
print(mean_Z)

np.savetxt('data/X_nt.csv',X, delimiter=',')
np.savetxt('data/Y_nt.csv',Y, delimiter=',')

import numpy as np
from scipy.stats import norm

def value_at_risk(X, z, epsilon):

    A = 0.025 * np.array([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]])
    B = 0.075 * np.array([[0,-1,-1],[-1,0,-1],[-1,-1,0],[0,-1,1],[-1,0,1],[-1,1,0],[0,1,-1],[1,0,-1],[1,-1,0],[0,1,1],[1,0,1],[1,1,0]])

    # since calling from Julia, X not a numpy array, need to transform it
    X_np = np.asarray(X)
    # do same for z even if not required, for code stability
    z_np = np.asarray(z)

    m=np.empty_like(z)
    a=np.empty_like(z)
    b=np.empty_like(z)
    for i in range(np.size(z)):
        m[i]=np.matmul(A[i],X_np[:,None])*z_np[i]
        a[i]=sum(A[i])*z[i]/4
        b[i]=np.matmul(B[i],X_np[:,None])*z_np[i]

    mean_profit=sum(m)
    var_profit=sum(np.square(a)) + sum(np.square(b))
    std_profit=np.sqrt(var_profit)

    value_at_risk_profit = mean_profit + std_profit * norm.ppf(epsilon)

    # loss VaR is negative of profit VaR
    return -value_at_risk_profit


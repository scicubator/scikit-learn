"""
LOST Example: Blind Source Separation of Laplacian Sources.

"""
print (__doc__)

import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from sklearn.decomposition import Lost

def sparse_signal(N, T=20000, density=.1):
  """Construct a sparse signal by specifying the `density` (percentage) of zeros 
  in the signal, where non-zero values are drawn from a laplacian.
  """
  n_non_zero = int(T*density)
  n_zero = T - n_non_zero
  Z = np.empty((N,T))
  for i in xrange(N):        
    Z[i,:] = np.random.permutation([0]* n_zero + 
    [np.random.laplace() for _ in xrange(n_non_zero)])
  return Z

def array_almost_equal_up_to_permutation_and_scaling(A, A_est, precision_places = 2):
  """A ~= A_est up to permutation and scaling of columns.
  
  Example
  -------
  
  >>> A = np.random.randint(0,10,(5,5))
  >>> array_almost_equal_up_to_permutation_and_scaling(A, -(A[:,::-1]+.001), 2)
  [-4, -3, -2, -1, 0]
  >>> array_almost_equal_up_to_permutation_and_scaling(A, (A[:,::-1]+.001), 1)
  [4, 3, 2, 1, 0]
  >>> array_almost_equal_up_to_permutation_and_scaling(A, -(A[:,::-1]+.01), 2)
  False
    
  """
  # Convert to matrices 
  if isinstance(A, np.ndarray):
    A = np.asmatrix(A)
  if isinstance(A_est, np.ndarray):
    A_est = np.asmatrix(A_est)    
 
  # Set precision
  precision = lambda X, prec : np.trunc(X * 10**prec) / 10.0**prec
  A = precision(A, precision_places)
  A_est = precision(A_est, precision_places) 
  
  # Compare columns
  N = A.shape[1]
  inds = [None] * N # Indices of the cols of A_est that match A.
  for i in xrange(N):
    for j in xrange(N):
        if all(A[:,i]-A_est[:,j] == 0):
          inds[i]=j
        if all(A[:,i]+A_est[:,j] == 0):
          inds[i]=-j
  # Check match for i-th column
  if None in inds:
    return False
        
  return inds
  
# Specify parameters  
T = 20000
max_iter = 50

# Mixing matrix
#A = np.array([[0.8321, 0.6247, -0.9939], [-0.5547, 0.7809, 0.1104]])
A = np.array([[0.8321, 0.6247], [-0.5547, 0.7809]])
M, N = A.shape

# Laplacian Sources
S = sparse_signal(N, T, density=.1)

# Observations
X=A.dot(S)

plt.scatter(X[0,:], X[1,:], marker='.')
plt.axis('equal')
plt.xlabel("X[0]")
plt.ylabel("X[1]")
plt.title("Scatterplot of X[0] V X[1]")
plt.show()

# Perform LOST!
lost = Lost(n_sources=N+20, max_iter=max_iter)
lost.fit(X)


print "A Estimate:"
print lost.A_  
 
print "Beta:", lost.beta_

assert array_almost_equal_up_to_permutation_and_scaling(A, np.array(lost.A_), precision_places=1) , \
  "Estimate is wrong."  

#S_est = lost.transform(X)
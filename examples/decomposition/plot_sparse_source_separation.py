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
  
  >>> A = np.random.random((5,5))
  >>> array_almost_equal_up_to_permutation_and_scaling(A, -A[:,::-1]+.01, 2)
  True
  >>> array_almost_equal_up_to_permutation_and_scaling(A, -A[:,::-1]+.1, 2)
  False
    
  """
  N = A.shape[1]
  for i in xrange(N):
    for j in xrange(N):
      try:
        assert_array_almost_equal(A[:,i], A_est[:,j], precision_places)
        break
      except AssertionError:
        try:
          assert_array_almost_equal(A[:,i], -A_est[:,j], precision_places)
          break
        except AssertionError:
          if j==N-1:
            return False
  return True
  
# Specify parameters  
T = 20000
max_iter = 50
#beta = 60

# Mixing matrix
A = np.array([[0.8321, 0.6247, -0.9939], [-0.5547, 0.7809, 0.1104]])
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
#plt.show()

# Perform LOST!
lost = Lost(n_sources=N+20, max_iter=max_iter)
A_est = lost.fit(X)


print "A Estimate:"
print A_est  
 
print "Beta:", lost.beta_

assert array_almost_equal_up_to_permutation_and_scaling(A, np.array(A_est), 1) , \
  "Estimate is wrong."  

#lost.project()
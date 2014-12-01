import numpy as np
from numpy.testing import assert_array_almost_equal 
from nose.tools import (assert_almost_equal, assert_equals, assert_true, \
  assert_false)
from sklearn.decomposition import Lost
from sklearn.decomposition.lost import _indicator_function, _soft_assignment


def sparse_signal(N, T=20000, density=.1):
  """Construct a sparse signal by specifying the `density` (percentage) of zeros 
  in the signal, where non-zero values are drawn from a laplacian.
  """
  n_non_zero = int(T * density)
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
  
  
class TestLOST:
  
  @classmethod
  def setup_class(self):
    """
    """
    pass
  
  @classmethod
  def teardown_class(self):
    """
    """
    pass
  
  def test__indicator_function(self):
    """Test `_indicator_function` using a canonical basis where one column is 
    selected as `line_vec` and all columns are tested against it. 
    
    """
    scalar = 5
    dim = 10 
    test_ind = dim/2
    X = np.matrix(np.eye(dim))*scalar
    v = X[:,test_ind]/scalar # normalised
    dists = _indicator_function(X, v)

    assert_equals(dists[0, test_ind], 0, msg="Vectors are co-incident therefore "\
      "indicator should be 0")
    assert_equals(dists[0,:].sum(), scalar*(dim-1), msg="Vectors are orthogonal therefore "\
      "indicator should be 1")
      
      
  def test__soft_assignment(self):
    """Test `_soft_assignment` for different values of `beta`.
    """
    V = np.matrix(np.eye(2))
    X = np.matrix([[1, .75, .5, .25, 0],[0, .25, .5, .75, 1]])

    # beta = ~inf - Hard assignment  
    weights = _soft_assignment(X, V, np.iinfo(np.int32).max)
    assert_array_almost_equal(V, np.matrix(weights[:,[0,-1]]), err_msg="Hard assignment") 
    
    # beta = 0 - Equal assignment
    weights = _soft_assignment(X, V, 0)
    assert_array_almost_equal(weights, np.ones((2,5)) *.5, err_msg="Equal Assignment")
    
    # beta = 1 - soft assignment
    weights = _soft_assignment(X, V, 1)
    assert_array_almost_equal(weights, np.matrix(
      [[ 0.73105858, 0.62245933, 0.5, 0.37754067, 0.26894142],
       [ 0.26894142, 0.37754067, 0.5, 0.62245933, 0.73105858]]),
      err_msg="Soft Assignment")
      
  
  def test_LOST(self):
    """Test the operation of LOST using a simple example.  
    """
    # Specify parameters
    T = 20000
    max_iter = 20
    density = .1
    random_state = 42 # Consistent starting point

    np.random.seed(random_state)
    
    # Mixing matrix
    A = np.array([[0.8321, 0.6247, -0.9939], [-0.5547, 0.7809, 0.1104]])
    M, N = A.shape
    
    # Laplacian Sources
    S = sparse_signal(N, T, density)
    
    # Observations
    X=A.dot(S)
   
    # Perform LOST!
    lost = Lost(n_sources=N+2, max_iter=max_iter, random_state=random_state)
    A_est = lost.fit(X)
    
    assert_true(array_almost_equal_up_to_permutation_and_scaling(A, np.array(A_est), 2), \
    "LOST did not converge.")
    
if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


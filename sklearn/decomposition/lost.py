"""
Line Orientation Separation Technique - LOST
"""

__author__ = "Paul D. O'Grady"
__license__ = "BSD"
__author_email__ = "paul@red-lab.ie"
__version__ = "0.1"

# Notes: 
# * Variable names follow those presented in our journal paper: doi:10.1155/2008/784296
# * Unit tests: `/sklearn/decomposition/tests/test_lost.py`

import numpy as np
from scipy.sparse import rand as srand
from sklearn.preprocessing import normalize

# Define consts
MAX_BETA = 200
MAX_ITER = 100
DEFAULT_EIGENVALUE_THRESH = 0.2
PRUNE_LINES_ITER_START = 5


def _indicator_function(X, line_vec):
  """Indicates the similarity between the data, columns of `X`, and the 
  column vectors, `line_vec`, via scalar projection.
  
   q_{it} = \norm{\mathbf{x}(t) - (\mathbf{v}_i \cdot \mathbf{x}(t)) \, \mathbf{v}_i},
  
  Example
  -------
  
  >>> i_th = 1
  >>> X=np.eye(4)
  >>> v=X[:,i_th]
  >>> _indicator_function(X, v)
  matrix([[ 1.,  0.,  1.,  1.]]) 
  # `i_th` vector is co-incident with `v` all others are orthogonal.
  
  """
  # Note: (v.x)*v = (|x|cos(theta))*v - scalar projection of x on v.
  # Multiply each individual dot product by the line vector
  q = X - np.kron(line_vec.T.dot(X), np.matrix(line_vec))
  # Calculate the L2 norm to determine distance
  q = np.sqrt(np.multiply(q, q).sum(axis=0))
  return q
  
def _soft_assignment(X, V, beta):
  """Perform a *soft-assignment* of the data, columns of `X`, to each line, columns 
  of `V`.
  
  Example
  -------
  
  >>> X = np.matrix([[1, .75, .5, .25, 0],[0, .25, .5, .75, 1]])
  >>> V = np.matrix(np.eye(2))
  >>> _soft_assignment(X, V, 1)
  array([[ 0.73105858, 0.62245933, 0.5, 0.37754067, 0.26894142],
         [ 0.26894142, 0.37754067, 0.5, 0.62245933, 0.73105858]])
  
  """
  T = X.shape[1]
  M, n_sources = V.shape
  
  assert n_sources > 1, "There must be more than one line to perform" \
                        "soft assignment."
  Q = np.empty((n_sources, T)) 
  for i in xrange(n_sources):
    q = _indicator_function(X, V[:,i])
    Q[i,:] = np.exp(-beta*q)
  Q /= Q.sum(axis=0)    # Paper: \tilde{q}_{it}
  return Q
  
def _args_under_threshold(vals, threshold):
  """Returns, in ascending order, the indices of the items in `vals` that are 
  below the specified `threshold`.
  
  Example
  -------
  
  >>> _args_under_threshold([0.1, 3, 73, 4, 5, 2.1], 4) 
  [0, 1, 5]
  
  """
  return sorted([i for i, j in enumerate(vals) if j < threshold])

class Lost(object):
  """Line Orientation Separation Technique - LOST
  """

  def _prune_lines(self, principal_eig_vals, max_remove=1):
    """
    """
    inds_below_thresh = _args_under_threshold(principal_eig_vals, 
                                                self.eigenvalue_threshold)                        
    if inds_below_thresh:
      remove_inds = inds_below_thresh[:max_remove]
      self.A_ = self.A_[:,[i for i in range(self.n_sources) if i not in remove_inds]]
      self.n_sources -= len(remove_inds)
                            
    pass
  
  def _update_beta(self, variances):
    """
    """
    beta = 1/max(variances)
  
  def _initialise(self, random_state):
    """Initial estimate, normailse columns
    """
    np.random.seed(random_state)
    self.A_ = np.matrix(normalize(np.random.randn(self.M, self.n_sources), 
                          norm='l2', axis=0))
    
  def _weighted_covariance(self, X, weightings, ddof=1):
    """
    """
    return np.multiply(weightings, X).dot(X.T) / (weightings.sum() * (X.shape[0]-ddof))
  
  def __init__(self, n_sources, beta=1, max_iter=MAX_ITER, random_state=None, 
               centre_data=True, max_beta=MAX_BETA, eigenvalue_threshold = 
               DEFAULT_EIGENVALUE_THRESH):
    """Line Orientation Separation Technique - LOST.
    
    LOST identifies lines in a scatterplot that cross the origin using a 
    proceedure similar to Expectation Maximisation (EM), where..... 
    
    Lost can be used to perform Independent Component Analysis (ICA) and
    Blind Source Separation (BSS).
    
    .. warning:: Docs are a work in progress.     
    
    Arguments:
    ----------
    `n_sources`: 
    `beta` : 
    `max_iter` :
    `random_stat` : 
    
    Notes
    -----
    LOST is introduced in:
     
     O'Grady, et al. The LOST Algorithm: Finding Lines and Separating Speech 
     Mixtures. EURASIP Journal on Advances in Signal Processing. vol. 2008, 
     Article ID 784296, 17 pages, 2008, DOI:10.1155/2008/784296.
    
    Please cite as::
    
     @article{OGRADY-2008b,
      author    = {Paul D. O'Grady and Barak A. Pearlmutter},
      title     = {The {LOST} Algorithm: Finding Lines and Separating Speech Mixtures},
      journal = {EURASIP Journal on  Advances in Signal Processing},
      note = "Article ID 784296, 17 pages",
      year = 2008,
      volume = 2008,
      doi = {http://dx.doi.org/10.1155/2008/784296}

    """
    self.n_sources = n_sources
    self.beta_ = beta
    self.max_iter = max_iter
    self.random_state = random_state
    
    self.centre_data = centre_data
    self.max_beta = max_beta
    self.eigenvalue_threshold = eigenvalue_threshold
    
    self.FINISHED = False
    
  def fit(self, X):
    """
    """
    self.M, T = X.shape
    
    self._initialise(self.random_state)
    
    print "Working..."

    # Centre data
    if self.centre_data:
      X = X-np.matrix(X.mean(axis=1)).T
      
    # Main loop
    for iter_ in range(self.max_iter):
      A_prev = self.A_.copy()
      
      # Expectation step : Soft assignment of observations, X, to line estimates, A_.
      self.Q_ = _soft_assignment(X, self.A_, self.beta_)
      
      # Maximization Step: Calculate weighted covariance for each line and perform PCA
      new_beta_candidates = [1./self.max_beta]
      principal_eig_vals = []
      for i in xrange(self.n_sources):
        values, vectors = np.linalg.eigh(
            self._weighted_covariance(X, self.Q_[i,:])) 
        second_eigval_ind, first_eigval_ind = values.argsort()[-2:]
        self.A_[:,i] = vectors[:, first_eigval_ind]  
        new_beta_candidates.append(values[second_eigval_ind])
        principal_eig_vals.append(values[first_eigval_ind])
        
      # Display convergence
      print "Iter: %d, No. Lines: %d, convergence: %f" \
        % (iter_, self.n_sources, np.linalg.norm(self.A_-A_prev, ord='fro'))
        
      # Update beta parameter
      self.beta_ = 1. / max(new_beta_candidates)  
      
      # Remove extraneous lines
      if iter_ > PRUNE_LINES_ITER_START:
        self._prune_lines(principal_eig_vals)
      
      
    print "Done."
    self.FINISHED = True
    return self.A_
    
    
  def project(self):
    """Recover `S` from `X` using `A_`. 
    """
    if not self.FINISHED:
      raise Exception, "Cannot perform `project` unless `fit` is performed first."
    else:
      if self.M < self.n_sources:
        raise NotImplementedError, \
          "For the under-determined case (M<N) we use L1-norm minimisation, " \
          "which is yet to be implemented."
      elif self.M > self.n_sources:
        raise NotImplementedError, \
          "For the over-determined case (M>N) we use the Moore-Penrose pseudoinverse, " \
          "which is yet to be implemented."
      elif self.M == self.n_sources:
        raise NotImplementedError, \
          "For the even-determined case (M=N) we use a linear transformation, " \
          "which is yet to be implemented."

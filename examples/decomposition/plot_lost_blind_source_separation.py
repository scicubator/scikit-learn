"""
Blind Source Separation using LOST
"""

print (__doc__)

import numpy as np
from scipy.io import wavfile
from scipy.fftpack import rfft, fft
import matplotlib.pyplot as plt
from numpy.lib import stride_tricks
from sklearn.decomposition import Lost
from numpy.testing import assert_array_almost_equal

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
  
def overlap_signal(x, N, overlap=0):
  """Construct an overlapping signal.
  """
  strides = ((N-overlap)*x.itemsize, x.itemsize )
  shape = (1+(x.nbytes - N*x.itemsize)/strides[0], N)
  return stride_tricks.as_strided(x, shape=shape, strides=strides)

# Load speech sources
#fs, s1 = wavfile.read('/data/audio_data/voxforge_wavs/anonymous-20080406-zsg/wav/b0426.wav')
#_, s2 = wavfile.read('/data/audio_data/voxforge_wavs/anonymous-20080328-mmv/wav/rb-10.wav')
#_, s3 = wavfile.read('/data/audio_data/voxforge_wavs/anonymous-20080408-sqi/wav/a0575.wav')

fs, s1 = wavfile.read('/data/audio_data/sources/instantaneous/Source1.wav')
_, s2 = wavfile.read('/data/audio_data/sources/instantaneous/Source2.wav')
_, s3 = wavfile.read('/data/audio_data/sources/instantaneous/Source3.wav')

# Create normalised equal-length sources
end = min(s1.shape[0], s2.shape[0], s3.shape[0])
S = np.c_[s1[:end], s2[:end], s3[:end]].T / float(2**15)

# Perform Mixing
A = np.array([[0.8321, 0.6247, -0.9939], [-0.5547, 0.7809, 0.1104]])
M, N = A.shape

# Generate observations
X=A.dot(S)

# Create sparse representation using FFT
N = 512
overlap = 512-128
#transform  = lambda row : np.ravel(rfft(np.hamming(N)*overlap_signal(row, N, overlap), axis=1))
transform  = lambda row : np.ravel(np.real(fft(np.hamming(N)*overlap_signal(row, N, overlap), axis=1)))
X_trans = np.apply_along_axis(transform, axis=1, arr=X)


# Display scatterplot 
plt.scatter(X_trans[0], X_trans[1], marker='.')
plt.axis('equal')
plt.xlabel("X[0]")
plt.ylabel("X[1]") 
plt.title("Scatterplot of X[0] V X[1]")
plt.show()
#assert False
# Perform LOST!
max_iter = 40
lost = Lost(n_sources=3, max_iter=max_iter, prune_lines=False, beta = 10)
lost.fit(X_trans)

print "**Results**"
print "A Estimate:"
print lost.A_  
print "Beta:", lost.beta_ or lost.beta
print

assert array_almost_equal_up_to_permutation_and_scaling(A, np.array(lost.A_), 1) , \
  "Estimate is wrong."  


# Plot sources and mixtures
plt.figure()
waveforms = [S[0,:], S[1,:], S[2,:], X[0,:], X[1,:]]
names = ['s1', 's2', 's3', 'x1', 'x2']
for i, (waveform, name) in enumerate(zip(waveforms, names)):
  plt.subplot(len(waveforms), 1, i+1)
  plt.plot(waveform)
  plt.title(name)
plt.tight_layout()
plt.show()

# XXX Note: need to implement L1-norm minimistion to recover sources

"""
LOST (Version 1.01) Log File

Sound File Type: wav
Mixture File Prefix: Mixture
Source Estimate File Prefix: LOST_SourceEstimate

Number of Mixtures = 2
Number of Sources = 3
Beta: Soft Boundary Value = 10.82708755
Convergence Precision = 0.0000100000000000
Convergence Iterations Limit = 90
Convergence Iterations = 8
FFT Points = 512
FFT Advance = 128

AEst = 
[0.62012 0.99391 0.83532]
[0.78451 -0.11015 -0.54976]

---
lost.beta_ = 30!



"""


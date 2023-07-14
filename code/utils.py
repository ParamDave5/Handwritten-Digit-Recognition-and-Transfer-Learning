import numpy as np


def flatten(X):
    # X : N x H x W
    X = X.reshape(X.shape[0], -1)
    return X

def y_encoding(y,M):
    N = y.shape[0]
    y_matrix = np.zeros((N,M))
    for n in range(N):
        y_matrix[n, y[n]] = 1
    return y_matrix

def PCA_proj(X):
    print('---------------- Performing PCA on Data --------------------')
    # X: N x D
    N, D = X.shape
    print(N,D)
    original_mean_img = np.mean(X, axis=0)
    X = X - np.mean(X, axis=0)
    sigma = 0
    sigma = (N-1)/ N * np.cov(X, rowvar = False)
    print('Covariance calculated')
    W, V = np.linalg.eig(sigma)
    keep_variance = 0.99 
    required_variance = keep_variance*sum(W)
    req_dim = 0
    variance = 0
    for i in range(len(W)):
      variance += np.abs(W[i])
      if variance >= required_variance:
          req_dim = i + 1
          m = req_dim
          break
    print('Required Dimension: ', m)
    idx = np.argsort(np.real(W))[::-1]
    V = V[:,idx]
    Updated_eigenvectors = V[:, :m]
    print('Got new clipped projection matrix')
    Updated_X = np.real(np.matmul(X, Updated_eigenvectors))
    print('X_train projected')
    return Updated_X, Updated_eigenvectors



import numpy as np

def calculate_MLE(x,y):
    mu = []
    sigma = []
    sigma_inv = []
    sigma_det = []
    threshold = 0.0000001
    for i in set(y):
        idx = np.where(y == i)
        mu_i = np.mean(x[idx], axis=0)
        sigma_i = np.cov(x[idx], rowvar=False)*(1/2)
        mu.append(mu_i)
        if np.linalg.det(sigma_i) < 0.00001:
            w, v = np.linalg.eig(sigma_i)
            sigma_det.append(np.product(np.real(w[w > threshold])))
            sigma_i = sigma_i + 0.0001*np.eye(len(mu_i))
            sigma.append(sigma_i)
            sigma_inv.append(np.linalg.inv(sigma_i))
        else:
            sigma.append(sigma_i)
            sigma_det.append(np.linalg.det(sigma_i))
            sigma_inv.append(np.linalg.inv(sigma_i))

    return mu, sigma, sigma_det, sigma_inv

def MDA(x,y,num_classes):
    n_classes = num_classes
    prior = 1/n_classes
    features = x.shape[1]
    mu,sigma,_,_ = calculate_MLE(x,y)
    mu = np.array(mu)
    anchor_mean = np.sum(prior*mu,axis = 0).reshape((1,features))
    sigma_b = np.zeros((features,features))
    sigma_w = np.zeros((features,features))
    for i in range(n_classes):
        sigma_b += prior*np.matmul((mu[i]-anchor_mean).T, mu[i]- anchor_mean)
        sigma_w += prior*sigma[i]
    sigma_w += 0.0001*np.eye(features)
    V, W = np.linalg.eig(np.matmul(np.linalg.inv(sigma_w), sigma_b))
    idx = np.argsort(np.real(V))[::-1]
    sorted_eigenvectors = W[:,idx]
    non_zero = np.count_nonzero(np.real(V)>1e-10)
    A = sorted_eigenvectors[:,0:non_zero]
    theta = (1/features)*A
    z = (np.matmul(theta.T, x.T)).T
    return np.real(z), theta



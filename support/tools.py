import torch
from sklearn.svm import SVC
import numpy as np
import math
from numpy.linalg import matrix_rank
import pickle
import os

def NTK_torch(X,d_max,fix_dep=0):
    K = torch.zeros((d_max, X.shape[0], X.shape[0]))
    S = torch.matmul(X, X.T)
    H = torch.zeros_like(S)
    for dep in range(d_max):
        if fix_dep <= dep:
            H += S
        K[dep] = H
        L = torch.diag(S)
        P = torch.clip(torch.sqrt(torch.outer(L, L)), a_min = 1e-9, a_max = None)
        Sn = torch.clip(S / P, a_min = -1, a_max = 1)
        S = (Sn * (torch.pi - torch.arccos(Sn)) + torch.sqrt(1.0 - Sn * Sn)) * P / 2.0 / torch.pi
        H = H * (torch.pi - torch.arccos(Sn)) / 2.0 / torch.pi
    return K[d_max - 1]

def NTK(X, d_max, fix_dep=0, normalize=False):
    K = np.zeros((d_max, X.shape[0], X.shape[0]))
    S = np.matmul(X, X.T)
    H = np.zeros_like(S)
    for dep in range(d_max):
        if fix_dep <= dep:
            H += S

        L = np.diag(S)
        P = np.clip(np.sqrt(np.outer(L, L)), a_min = 1e-9, a_max = None)
        Sn = np.clip(S / P, a_min = -1, a_max = 1)
        S = (Sn * (math.pi - np.arccos(Sn)) + np.sqrt(1.0 - Sn * Sn)) * P / 2.0 / math.pi
        H = H * (math.pi - np.arccos(Sn)) / 2.0 / math.pi
        K[dep] = H
    ntk = K[d_max - 1]
    if normalize:
        ntk = normalize_kernel(ntk)
    return ntk
def svm(K_train, K_test, y_train, C):
    # n_val, n_train = K2.shape
    clf = SVC(kernel = "precomputed", C = C, cache_size = 100000)
    clf.fit(K_train, y_train)
    return clf.predict(K_test)

def normalize_kernel(K:np.ndarray):
    assert len(K.shape) == 2 and K.shape[0] == K.shape[1]
    assert (K.diagonal() == 0).sum() == 0 # no zero diagonal entry
    k = 1./np.sqrt(K.diagonal())
    inv_norm = np.outer(k,k)
    return K*inv_norm

def transpose_values(d):
    new_dict = {}
    for key,value in d.items():
        new_dict[key] = value.transpose()
    return new_dict

def ridge_reg(M,coef=None, return_coef=False):
    assert len(M.shape)==2 and M.shape[0] == M.shape[1]
    if coef is None:
        coef,ridge_M = choose_ridge_coef(M)
    else:
        ridge_M = M + coef*np.eye(M.shape[0])
    if return_coef:
        return ridge_M, coef
    else:
        return ridge_M

def choose_ridge_coef(M):
    assert len(M.shape)==2 and M.shape[0] == M.shape[1]
    if matrix_rank(M)==M.shape[0]:
        return 0,M
    coef = np.maximum(1e-10,np.abs(np.diagonal(M)).min()/100)
    ridge_M = ridge_reg(M,coef)
    while matrix_rank(ridge_M)< M.shape[0]:
        coef*=2
        ridge_M = ridge_reg(M,coef)
    return coef,ridge_M



def reshape_cat(Ms,shape,return_idxes=True):
    output = []
    idxes=[]
    cur = 0
    for M in Ms:
        new_M = M.reshape(*shape)
        output.append(new_M)
        idx = np.arange(cur,cur+len(new_M))
        idxes.append(idx.reshape(M.shape[:2]))
        cur += len(new_M)

    if return_idxes:
        return np.concatenate(output,axis=0),idxes
    else:
        return np.concatenate(output,axis=0)


def check_if_values_unchanged(old_dict,new_dict,keys):
    for key in keys:
        assert key in old_dict and key in new_dict
        old_value = old_dict[key]
        new_value = new_dict[key]
        if isinstance(old_value,list) or isinstance(old_value,np.ndarray):
            for v1,v2 in zip(old_value,new_value):
                if v1 != v2:
                    return False
        else:
            if old_value != new_value:
                return False
    return True

def load_cifar(path="cifar-10-batches-py"):
    train_batches = []
    train_labels = []

    for i in range(1, 6):
        cifar_out = pickle.load(open(os.path.join(path, "data_batch_{0}".format(i)),'rb'),encoding='bytes')
        train_batches.append(cifar_out[b"data"])
        train_labels.extend(cifar_out[b"labels"])
    X_train = np.vstack(tuple(train_batches)).reshape(-1, 3, 32, 32)
    y_train = np.array(train_labels)

    cifar_out = pickle.load(open(os.path.join(path, "test_batch"),'rb'),encoding='bytes')
    X_test = cifar_out[b"data"].reshape(-1, 3, 32, 32)
    y_test = cifar_out[b"labels"]

    X_train = (X_train / 255.0).astype(np.float32)
    X_test = (X_test / 255.0).astype(np.float32)
    mean = X_train.mean(axis=(0, 2, 3))
    std = X_train.std(axis=(0, 2, 3))
    X_train = (X_train - mean[:, None, None]) / std[:, None, None]
    X_test = (X_test - mean[:, None, None]) / std[:, None, None]

    return (X_train, np.array(y_train)), (X_test, np.array(y_test))

def one_hot_label(Ys,n_class):
    Ys_onehot = []
    for Y in Ys:
        N = len(Y)
        Y_onehot= np.ones((N, n_class)) * (-1 / n_class)
        for i in range(N):
            Y_onehot[i][Y[i]] = 1 - 1 / n_class
        Ys_onehot.append(Y_onehot)
    return np.array(Ys_onehot)

def pred_from_one_hot(pred):
    return np.argmax(pred,axis=1)


def L2_loss(pred,label):
    return np.mean((pred-label)**2)
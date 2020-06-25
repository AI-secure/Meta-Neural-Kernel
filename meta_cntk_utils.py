import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from time import time
from numpy.linalg import matrix_rank
from numpy.linalg import pinv,inv
from numpy.linalg import eig as eig
from numpy.linalg import eigh,lstsq
from numpy.linalg import matrix_power
from scipy.linalg import expm,pinvh,solve
from tqdm.notebook import tqdm,trange
from support.omniglot_loaders import OmniglotNShot
from support.tools import *
from sklearn import decomposition
from scipy.spatial.distance import pdist,squareform
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from meta_cntk import MetaCNTK
import argparse
from types import SimpleNamespace
from sklearn.decomposition import PCA,IncrementalPCA,KernelPCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from collections import deque
import argparse
from time import time
import typing

import pandas as pd
import matplotlib as mpl


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import higher
import pickle
from tqdm.notebook import tqdm,trange
from support.omniglot_loaders import OmniglotNShot


def load_dataset(n_task,random=True,seed=0,load_embeddings=False):
    # Load preprocesses Ominiglot dataset for 20-way 1-shot classification
    # path = f'saved_models/tasks-{n_task}.p'
    # if os.path.exists(path):
    #     print(f"Dataset exists at {path}")
    #     tasks = pickle.load(open(path,'rb'))
    # else:
    tasks = pickle.load(open(f'saved_models/tasks-200.p', 'rb'))
    # Get the subset of the size we need

    n_all_tasks = len(tasks['X_qry'])
    assert n_all_tasks == 200
    assert n_task <= n_all_tasks
    if random:
        np.random.seed(seed)
        idxes = np.random.choice(n_all_tasks, size=n_task, replace=False)
    else:
        idxes = np.arange(n_task)
    tasks['X_qry'] = tasks['X_qry'][idxes]
    tasks['X_spt'] = tasks['X_spt'][idxes]
    tasks['Y_qry'] = tasks['Y_qry'][idxes]
    tasks['Y_spt'] = tasks['Y_spt'][idxes]
    tasks['idx_Xs']= tasks['idx_Xs'][idxes]
    tasks['idx_Xs_'] = tasks['idx_Xs_'][idxes]
    tasks['n_task'] = n_task

    # if load_embeddings:
    #     embeddings = load_label_embeddings(200,random_cnn=random_cnn_embedding)
    #     tasks['Y_qry_emb'] = embeddings['Y_qry_emb'][idxes]
    #     tasks['Y_spt_emb'] = embeddings['Y_spt_emb'][idxes]
    #     tasks['test_Y_qry_emb'] = embeddings['test_Y_qry_emb']
    #     tasks['test_Y_spt_emb'] = embeddings['test_Y_spt_emb']
    tasks['load_embeddings'] = load_embeddings

    return SimpleNamespace(**tasks)


def load_precomputed_base_kernels(dataset,kernel='CNTK'):
    # path = f'saved_models/CNTK-{n_task}.npy'
    # if os.path.exists(path):
    #     print(f"Precomputed CNTK exists at {path}")
    #     CNTK = np.load(path)
    # else:
    CNTK_all = np.load(f'saved_models/CNTK-200.npy')
    all_idxes = np.concatenate([dataset.idx_Xs.flatten(),
                                dataset.idx_Xs_.flatten(),
                                dataset.idx_test_Xs.flatten(),
                                dataset.idx_test_Xs_.flatten()])
    dataset.all_idxes = all_idxes
    dataset.CNTK = CNTK_all[all_idxes][:, all_idxes]

def load_label_embeddings(n_task,random_cnn=False):
    #     postfix = f'-{emb_method}' if emb_method != '' else ''
    if not random_cnn:
        path = f'saved_models/tasks-{n_task}-embeddings.p'
    else:
        path = f'saved_models/tasks-{n_task}-embeddings-random_cnn.p'
    embedding_dict = pickle.load(open(path, 'rb'))
    return embedding_dict


def get_embeddings_from_PCA(dataset, n_components=784, PCA_method='regular'):
    # 784 = 28*28, which is the number of pixels in each original image.
    # It is also the maximum n_componetns for PCA that we can choose

    X_qry = dataset.X_qry if not dataset.load_embeddings else dataset.Y_qry_emb
    X_spt = dataset.X_spt if not dataset.load_embeddings else dataset.Y_spt_emb
    test_X_qry = dataset.test_X_qry if not dataset.load_embeddings else dataset.test_Y_qry_emb
    test_X_spt = dataset.test_X_spt if not dataset.load_embeddings else dataset.test_Y_spt_emb
    if PCA_method == 'regular':
        pca = PCA(n_components=n_components, svd_solver='randomized')
    else:
        assert PCA_method in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
        pca = KernelPCA(n_components=n_components, kernel=PCA_method, fit_inverse_transform=True)

    if not dataset.load_embeddings:
        # Reshape images from vectors to their original size: 32*32
        new_shape = (-1, 32, 32)
        X_train = np.concatenate([dataset.X_qry.reshape(new_shape), dataset.X_spt.reshape(new_shape)], axis=0)
        X_train = X_train[:, 2:-2, 2:-2] # remove paddings
        pca.fit(X_train.reshape(X_train.shape[0], -1))
    else:
        emb_dim = X_qry.shape[-1]
        X_train = np.concatenate([X_qry.reshape(-1,emb_dim),
                                  X_spt.reshape(-1,emb_dim)], axis=0)
        pca.fit(X_train)
        # print('X_train',X_train.shape)
    Xs = [X_qry, X_spt, test_X_qry, test_X_spt]
    Y_qry_emb, Y_spt_emb, test_Y_qry_emb, test_Y_spt_emb = [], [], [], []
    Ys = [Y_qry_emb, Y_spt_emb, test_Y_qry_emb, test_Y_spt_emb]
    for x, y in zip(Xs, Ys):
        # The following 3 lines are to remove the padding in original images (28*28 pixels),
        # since we pad the original images to 32*32 for convenience of CNTK computing via CUDA
        if not dataset.load_embeddings:
            x = x.reshape(-1, 32, 32)
            x = x[:, 2:-2, 2:-2]
            x = x.reshape(x.shape[0], -1)
        else:
            x = x.reshape(-1,emb_dim)
        result = pca.transform(x)
        y.append(result)
    dataset.Y_qry_emb, dataset.Y_spt_emb, dataset.test_Y_qry_emb, dataset.test_Y_spt_emb = Y_qry_emb[0], Y_spt_emb[0], test_Y_qry_emb[0], \
                                                           test_Y_spt_emb[0]
    # return SimpleNamespace(Y_qry_emb=Y_qry_emb, Y_spt_emb=Y_spt_emb,
    #                        test_Y_qry_emb=test_Y_qry_emb, test_Y_spt_emb=test_Y_spt_emb)


def preprocess_label_embeddings(dataset,pred_test_Y_qry=None,test_all = False):
    # Find the center of embeddings in each class, then use this center as the label for this class
    n_components = dataset.Y_qry_emb.shape[-1]

    Y_qry_emb = dataset.Y_qry_emb.reshape(*dataset.Y_qry.shape, n_components)
    Y_spt_emb = dataset.Y_spt_emb.reshape(*dataset.Y_spt.shape, n_components)
    test_Y_qry_emb = dataset.test_Y_qry_emb.reshape(*dataset.test_Y_qry.shape, n_components)
    test_Y_spt_emb = dataset.test_Y_spt_emb.reshape(*dataset.test_Y_spt.shape, n_components)

    # Y_qry_emb,Y_spt_emb
    clf = NearestCentroid()
    Y_train = np.concatenate([dataset.Y_qry, dataset.Y_spt], axis=1)
    Y_train_emb = np.concatenate([Y_qry_emb, Y_spt_emb], axis=1)
    N_train = len(Y_train)
    n_class = len(np.unique(Y_train[0]))
    Y_centroids = []

    for i in range(N_train):
        clf.fit(Y_train_emb[i], Y_train[i])
        for j in range(n_class):
            Y_train_emb[i][Y_train[i] == j] = clf.centroids_[j]
        centroids = clf.centroids_
        Y_centroids.append(centroids)
    Y_qry_emb = Y_train_emb[:, :dataset.Y_qry.shape[1], :]
    Y_spt_emb = Y_train_emb[:, dataset.Y_qry.shape[1]:, :]
    Y_centroids = np.array(Y_centroids)

    # Y_qry_emb,Y_spt_emb
    clf = NearestCentroid()
    Y_test = np.concatenate([dataset.test_Y_qry, dataset.test_Y_spt], axis=1)
    Y_test_emb = np.concatenate([test_Y_qry_emb, test_Y_spt_emb], axis=1)

    if pred_test_Y_qry is not None:
        pred_Y_test = np.concatenate([pred_test_Y_qry,dataset.test_Y_spt],axis=1)

    N_test = len(Y_test)
    n_class = len(np.unique(Y_test[0]))
    test_Y_centroids = []

    for i in range(N_test):
        if pred_test_Y_qry is None:
            Y_emb = test_Y_spt_emb[i]
            Y = dataset.test_Y_spt[i]
            clf.fit(Y_emb, Y)
        else:
            clf.fit(Y_test_emb[i], pred_Y_test[i])

            for j in range(n_class):
                nbrs = NearestNeighbors(n_neighbors=1)
                embs = Y_test_emb[i][pred_Y_test[i] == j]
                nbrs.fit(embs)
                _,[[emb_idx]] = nbrs.kneighbors([clf.centroids_[j]])
                clf.centroids_[j]=embs[emb_idx]

        for j in range(n_class):
            Y_test_emb[i][Y_test[i] == j] = clf.centroids_[j]
        centroids = clf.centroids_
        test_Y_centroids.append(centroids)
    test_Y_qry_emb = Y_test_emb[:, :dataset.test_Y_qry.shape[1], :]
    test_Y_spt_emb = Y_test_emb[:, dataset.test_Y_qry.shape[1]:, :]
    test_Y_centroids = np.array(test_Y_centroids)

    dataset.Y_qry_emb=Y_qry_emb
    dataset.Y_spt_emb=Y_spt_emb
    dataset.test_Y_qry_emb=test_Y_qry_emb
    dataset.test_Y_spt_emb=test_Y_spt_emb
    dataset.Y_centroids=Y_centroids
    dataset.test_Y_centroids=test_Y_centroids
    dataset.n_components=n_components
    dataset.N_train = N_train
    dataset.N_test = N_test

def pred_from_emb(embeddings, dataset, n_neighbors=1):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    assert len(embeddings) == len(dataset.test_Y_centroids)

    preds = []
    for i in range(dataset.N_test):
        nbrs.fit(dataset.test_Y_centroids[i])
        emb = embeddings[i]
        _, pred = nbrs.kneighbors(emb)
        pred = pred.flatten()
        preds.append(pred)
    preds = np.array(preds)
    return preds



def build_MetaCNTK(dataset, ridge_coef=[1e-5, 1e-5], normalize_NTK=True, normalize_metaNTK=True):

    model = MetaCNTK(d_max=20, fix=False, GAP=True,
                     inner_lr=np.inf, train_time=np.inf,
                     invMetaNTK=False,
                     kernel_ridge=True,
                     ridge_coef=ridge_coef,
                     normalize_NTK=normalize_NTK,
                     normalize_metaNTK=normalize_metaNTK)
    model.fit(dataset.X_qry,dataset.Y_qry_emb,dataset.X_spt,dataset.Y_spt_emb)
    model.load_test_tasks(X_query=dataset.test_X_qry,X_support=dataset.test_X_spt,Y_support=dataset.test_Y_spt_emb)
    model.load_precompute_NTKs(dataset.CNTK)
    return model

def test_MetaCNTK(dataset,model):
    t0 = time()
    pred_test_Y = model.predict()
    print(f"Took {round(time() - t0, 2)}")
    loss = np.mean( (pred_test_Y - dataset.test_Y_qry_emb)**2)
    pred_test_Y = pred_from_emb(pred_test_Y, dataset)

    pred_test_Y = pred_test_Y.reshape(*dataset.test_Y_qry.shape)
    test_acc = np.mean(pred_test_Y == dataset.test_Y_qry)
    return test_acc, pred_test_Y, loss

def augment_train_data(dataset,enlarge_ratio=10,n_way=5,n_shot=1,seed=0):
    new_n_task =dataset.n_task*enlarge_ratio
    X = np.concatenate([dataset.X_qry,dataset.X_spt],axis=1)
    Y = np.concatenate([dataset.Y_qry,dataset.Y_spt],axis=1)
    idx_X = np.concatenate([dataset.idx_Xs,dataset.idx_Xs_],axis=1)

    dict_idx_x = {}
    for i in range(idx_X.shape[0]):
        for j in range(idx_X.shape[1]):
            idx = idx_X[i][j]
            x = X[i][j]
            dict_idx_x[idx] = x

    n_local_labels = len(np.unique(Y))
    n_global_labels = 0
    for i in range(Y.shape[0]):
        Y[i] += n_global_labels
        n_global_labels += n_local_labels

    global_labels = np.unique(Y)

    Y = Y.flatten()
    idx_X = idx_X.flatten()

    dict_label_idx = {}
    dict_idx_label = {}
    for label in global_labels:
        idxes_for_label = idx_X[Y == label]
        dict_label_idx[label] = idxes_for_label
        for idx in idxes_for_label:
            dict_idx_label[idx] = label

    X_qry,X_spt,Y_spt,Y_qry,idx_X_qry,idx_X_spt = [],[],[],[],[],[]
    np.random.seed(seed)
    all_labels = np.concatenate([np.random.choice(global_labels, size=len(global_labels), replace=False) for _ in
                                 range(enlarge_ratio)]).reshape(-1, n_way)
    assert len(all_labels) == new_n_task
    for i_task in range(new_n_task):
        # labels = np.random.choice(global_labels,size = n_way,replace=False)
        labels = all_labels[i_task]
        idx_X_qry.append([]),idx_X_spt.append([])
    #     Y_qry.append([]),Y_spt.append([])
        for label in labels:
    #         print(labels)
            idx_spt,idx_qry = train_test_split(dict_label_idx[label],train_size = n_shot)
            idx_X_qry[-1].append(idx_qry)
            idx_X_spt[-1].append(idx_spt)

    idx_X_qry = np.array(idx_X_qry).reshape(len(idx_X_qry),-1)
    idx_X_spt = np.array(idx_X_spt).reshape(len(idx_X_spt),-1)



    Y_qry_emb,Y_spt_emb,test_Y_qry_emb,test_Y_spt_emb = [],[],[],[]

    for idx in idx_X_qry.flatten():
        Y_qry.append(dict_idx_label[idx])
        X_qry.append(dict_idx_x[idx])
        Y_qry_emb.append(dataset.dict_idx_emb[idx])

    for idx in idx_X_spt.flatten():
        Y_spt.append(dict_idx_label[idx])
        X_spt.append(dict_idx_x[idx])
        Y_spt_emb.append(dataset.dict_idx_emb[idx])

    x_shape = X_spt[0].shape
    emb_shape = Y_spt_emb[0].shape
    Y_qry,Y_spt = np.array(Y_qry),np.array(Y_spt)
    Y_qry_emb,Y_spt_emb = np.array(Y_qry_emb), np.array(Y_spt_emb)
    X_qry,X_spt = np.array(X_qry),np.array(X_spt)


    Y_qry,Y_spt = Y_qry.reshape(idx_X_qry.shape),Y_spt.reshape(idx_X_spt.shape)
    Y_qry_emb,Y_spt_emb = Y_qry_emb.reshape(idx_X_qry.shape+emb_shape), Y_spt_emb.reshape(idx_X_spt.shape+emb_shape)
    X_qry,X_spt = X_qry.reshape(idx_X_qry.shape + x_shape),X_spt.reshape(idx_X_spt.shape+x_shape)

    from copy import deepcopy
    np.random.seed(seed)
    for i in range(len(Y_qry)):
        ys_qry = deepcopy(Y_qry[i])
        ys_spt = deepcopy(Y_spt[i])
        label_mapping = {}
        labels = np.unique(ys_qry)

        new_labels = np.arange(n_way)

        np.random.shuffle(new_labels)

        for label,new_label in zip(labels,new_labels):

            Y_qry[i][ys_qry==label] = new_label
            Y_spt[i][ys_spt==label] = new_label

    dataset.idx_Xs = idx_X_qry
    dataset.idx_Xs_ = idx_X_spt
    dataset.X_qry = X_qry
    dataset.X_spt = X_spt
    dataset.Y_qry = Y_qry
    dataset.Y_spt = Y_spt
    dataset.Y_qry_emb = Y_qry_emb
    dataset.Y_spt_emb = Y_spt_emb




def train_supervised(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def test_supervised(model, device, test_loader, verbose=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return test_loss,test_acc

class NaiveDataset(torch.utils.data.Dataset):
    def __init__(self, samples,labels):
        'Initialization'
        self.labels = torch.from_numpy(labels).long()
        self.samples = torch.from_numpy(samples).float()
        assert len(labels) == len(samples)
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        X = self.samples[index]
        y = self.labels[index]
        return X, y

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def build_CNN(n_way,device,n_channel=64,batch_norm = True,dropout=None):
    if dropout == 0:
        dropout = None
    modules = [nn.Conv2d(1, n_channel, 3),
            nn.BatchNorm2d(n_channel, momentum=1, affine=True) if batch_norm else None,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout) if dropout is not None else None,
            nn.Conv2d(n_channel, n_channel, 3),
            nn.BatchNorm2d(n_channel, momentum=1, affine=True) if batch_norm else None,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout) if dropout is not None else None,
            nn.Conv2d(n_channel, n_channel, 3),
            nn.BatchNorm2d(n_channel, momentum=1, affine=True) if batch_norm else None,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout) if dropout is not None else None,
            Flatten(),
            nn.Linear(n_channel, n_way)]
    for i,module in enumerate(modules):
        if module is None:
            del modules[i]
    net = nn.Sequential(*modules).to(device)
    net.eval()
    return net


def get_train_data(dataset, n_test_per_class=0):
    X = np.concatenate([dataset.X_qry, dataset.X_spt], axis=1)
    Y = np.concatenate([dataset.Y_qry, dataset.Y_spt], axis=1)
    idx_X = np.concatenate([dataset.idx_Xs, dataset.idx_Xs_], axis=1)

    dict_idx_x = {}
    for i in range(idx_X.shape[0]):
        for j in range(idx_X.shape[1]):
            idx = idx_X[i][j]
            x = X[i][j]
            dict_idx_x[idx] = x

    n_local_labels = len(np.unique(Y))
    n_global_labels = 0
    for i in range(Y.shape[0]):
        Y[i] += n_global_labels
        n_global_labels += n_local_labels

    global_labels = np.unique(Y)

    Y = Y.flatten()
    idx_X = idx_X.flatten()

    dict_label_idx = {}
    dict_idx_label = {}
    for label in global_labels:
        idxes_for_label = idx_X[Y == label]
        dict_label_idx[label] = idxes_for_label
        for idx in idxes_for_label:
            dict_idx_label[idx] = label
    labels = []
    samples = []
    for label, idxes in dict_label_idx.items():
        if n_test_per_class > 0:
            idxes = idxes[:-n_test_per_class]
        for idx in idxes:
            labels.append(label)
            samples.append(dict_idx_x[idx])

    samples = np.array(samples)
    labels = np.array(labels)
    if samples.shape[-1] == 32:  # remove useless padding
        samples = samples[:, :, 2:-2, 2:-2]
    train_set = {'samples': samples, 'labels': labels}
    n_class = len(dict_label_idx.keys())
    assert n_class == len(np.unique(train_set['labels']))
    assert np.max(train_set['labels']) == n_class - 1
    train_set['n_class'] = n_class
    if n_test_per_class > 0:
        labels = []
        samples = []
        for label, idxes in dict_label_idx.items():
            idxes = idxes[-n_test_per_class:]
            for idx in idxes:
                labels.append(label)
                samples.append(dict_idx_x[idx])
        samples = np.array(samples)
        labels = np.array(labels)
        if samples.shape[-1] == 32:  # remove useless padding
            samples = samples[:, :, 2:-2, 2:-2]
        test_set = {'samples': samples, 'labels': labels}

        return train_set, test_set
    else:
        return train_set,None


def pretrain(net,train_set, test_set, device, batch_size=64, lr=1e-3, epochs=40, seed=0,weight_decay=0.):
    if epochs == 0:
        return net
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = torch.utils.data.DataLoader(
        NaiveDataset(train_set['samples'], train_set['labels']),
        batch_size=batch_size, shuffle=True, **kwargs)
    if test_set is not None:
        test_loader = torch.utils.data.DataLoader(
            NaiveDataset(test_set['samples'], test_set['labels']),
            batch_size=batch_size, shuffle=True, **kwargs)

    optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=weight_decay)
    test_accs = []
    test_losses = []
    for epoch in trange(epochs, leave=False, desc='Train Supervised'):
        train_supervised(net, device, train_loader, optimizer)
        if test_set is not None:
            test_loss, test_acc = test_supervised(net, device, test_loader)
            test_accs.append(test_acc)
            test_losses.append(test_loss)
    if test_set is not None:
        return net,np.array(test_accs),np.array(test_losses)
    else:
        return net, None, None


def encode_labels(dataset,net,device):
    feature_extractor = net[:-1]
    Y_qry_emb,Y_spt_emb,test_Y_qry_emb,test_Y_spt_emb = [],[],[],[]
    for x,y in [(dataset.X_qry,Y_qry_emb),(dataset.X_spt,Y_spt_emb),
                (dataset.test_X_qry,test_Y_qry_emb),(dataset.test_X_spt,test_Y_spt_emb)]:
        x = x.reshape(-1,1,32,32)
        x = x[:,:,2:-2,2:-2]
        x = torch.from_numpy(x).to(device)
        x = x.reshape(( -1, 5 ,)+x.shape[1:]) # reshape into batches of size = 5 for memory efficiency
        result = []
        for batch_x in x:
            result.append(feature_extractor(batch_x).detach().cpu().numpy())
        result = np.concatenate(result,axis=0)
        y.append(result)

    Y_qry_emb,Y_spt_emb,test_Y_qry_emb,test_Y_spt_emb = Y_qry_emb[0],Y_spt_emb[0],test_Y_qry_emb[0],test_Y_spt_emb[0]
    emb_dim = Y_qry_emb.shape[-1]
    dict_idx_emb = {}
    for embs, idxes in [(Y_qry_emb, dataset.idx_Xs), (Y_spt_emb, dataset.idx_Xs_),
                        (test_Y_qry_emb, dataset.idx_test_Xs),
                        (test_Y_spt_emb, dataset.idx_test_Xs_)]:
        idxes = idxes.flatten()
        for emb, idx in zip(embs, idxes):
            dict_idx_emb[idx] = emb


    dataset.Y_qry_emb = Y_qry_emb.reshape(dataset.n_task,-1,emb_dim)
    dataset.Y_spt_emb = Y_spt_emb.reshape(dataset.n_task,-1,emb_dim)
    dataset.test_Y_qry_emb = test_Y_qry_emb.reshape(dataset.test_Y_qry.shape+(emb_dim,))
    dataset.test_Y_spt_emb = test_Y_spt_emb.reshape(dataset.test_Y_spt.shape+(emb_dim,))
    dataset.dict_idx_emb = dict_idx_emb
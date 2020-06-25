import cupy as cp
import numpy as np
import argparse
import scipy.linalg
from sklearn.svm import SVC
import os
import pickle
from tqdm import trange,tqdm
# from utils import load_cifar

# parser = argparse.ArgumentParser(description='Convolutional Neural Tangent Kernel (CNTK) for CIFAR-10')
# parser.add_argument('--depth', default=21, type=int, help='depth of CNTK (#conv layers + 1)')
# parser.add_argument('--gap', default="yes", type=str, help='whether GAP (global average pooling) is used')
# parser.add_argument('--fix', default="yes", type=str,
#                     help='whether first layer and last layer are fixed (or trained) (see Section 4.2 in our paper)')
# args = parser.parse_args()
#
# d = args.depth
# gap = (args.gap == "yes")
# fix = (args.fix == "yes")

# CUDA kernel for convolution operation
conv3 = cp.RawKernel(r'''
extern "C" __global__
void conv3(const float s[32][32][32][32], float t[32][32][32][32])
{
	int x1 = threadIdx.x + blockIdx.x - 31;
	int y1 = threadIdx.y + blockIdx.y - 31;
	int x2 = threadIdx.x;
	int y2 = threadIdx.y;
	__shared__ float d[32 + 2][32 + 2];
	if (x2 == 0){
		d[0][y2 + 1] = d[33][y2 + 1] = 0;
		if (x2 == 0 && y2 == 0)
			d[0][0] = d[0][33] = d[33][0] = d[33][33] = 0; 
	}
	if (y2 == 0){
		d[x2 + 1][0] = d[x2 + 1][33] = 0;
	}
	if (x1 < 0 || x1 > 31 || y1 < 0 || y1 > 31){
		d[x2 + 1][y2 + 1] = 0;
		return;
	}
	else
		d[x2 + 1][y2 + 1] = s[x1][y1][x2][y2];
	__syncthreads();
	t[x1][y1][x2][y2] = d[x2][y2] + d[x2][y2 + 1] + d[x2][y2 + 2]
					  + d[x2 + 1][y2] + d[x2 + 1][y2 + 1] + d[x2 + 1][y2 + 2]
					  + d[x2 + 2][y2] + d[x2 + 2][y2 + 1] + d[x2 + 2][y2 + 2];
}''', 'conv3')
conv_blocks = (63, 63)
conv_threads = (32, 32)

# CUDA kernel for activation
trans = cp.RawKernel(r'''
extern "C" __global__
void trans(float s[32][32][32][32], float t[32][32][32][32], const float l[32][32], const float r[32][32], const float il[32][32], const float ir[32][32])
{
	int x1 = blockIdx.x;
	int y1 = blockIdx.y;
	int x2 = threadIdx.x + ((blockIdx.z >> 2) << 3);
	int y2 = threadIdx.y + ((blockIdx.z & 3) << 3);
	float S = s[x1][y1][x2][y2], T = t[x1][y1][x2][y2], L = l[x1][y1], R = r[x2][y2], iL = il[x1][y1], iR = ir[x2][y2];
	S = S * iL * iR;
	float BS = (S * (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) + sqrtf(1.0f - min(S * S, 1.0f))) * L * R / 28.274333882308138f;
	S = (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) / 28.274333882308138;
	t[x1][y1][x2][y2] = T * S + BS;
	s[x1][y1][x2][y2] = BS;
}''', 'trans')
trans_blocks = (32, 32, 16)
trans_threads = (8, 8)


# Calculate diagonal entries of $\Sigma^{(h)}(x, x)$ and their reciprocals. See Section 4.3 in our paper.
def xx(x,d:int,fix:bool):
    RL = [1.0, ]
    iRL = [1.0, ]

    S = cp.matmul(x.T, x).reshape(32, 32, 32, 32)
    conv3(conv_blocks, conv_threads, (S, S))
    T = cp.zeros((32, 32, 32, 32), dtype=cp.float32)
    if not fix:
        T += S

    for i in range(1, d - 1):
        L = cp.sqrt(cp.diag(S.reshape(1024, 1024)).reshape(32, 32))
        iL = 1.0 / L
        RL.append(L)
        iRL.append(iL)
        trans(trans_blocks, trans_threads, (S, T, L, L, iL, iL))
        conv3(conv_blocks, conv_threads, (S, S))
        conv3(conv_blocks, conv_threads, (T, T))

    L = cp.sqrt(cp.diag(S.reshape(1024, 1024)).reshape(32, 32))
    iL = 1.0 / L
    RL.append(L)
    iRL.append(iL)
    trans(trans_blocks, trans_threads, (S, T, L, L, iL, iL))

    if fix:
        T -= S
    return RL, iRL


# Caclulate the kernel value of x and z.
# Lx and Lz are diagonal entries of $\Sigma^{(h)}(x, x)$ and $\Sigma^{(h)}(z, z)$.
# iLx and iLz are reciprocals of diagonal entries of $\Sigma^{(h)}(x, x)$ and $\Sigma^{(h)}(z, z)$.
def xz(x, z, Lx, Lz, iLx, iLz, d:int, fix:bool, gap:bool):
    S = cp.matmul(x.T, z).reshape(32, 32, 32, 32)
    conv3(conv_blocks, conv_threads, (S, S))
    T = cp.zeros((32, 32, 32, 32), dtype=cp.float32)
    if not fix:
        T += S

    for i in range(1, d - 1):
        trans(trans_blocks, trans_threads, (S, T, Lx[i], Lz[i], iLx[i], iLz[i]))
        conv3(conv_blocks, conv_threads, (S, S))
        conv3(conv_blocks, conv_threads, (T, T))

    trans(trans_blocks, trans_threads, (S, T, Lx[-1], Lz[-1], iLx[-1], iLz[-1]))

    if fix:
        T -= S
    return cp.mean(T) if gap else cp.trace(T.reshape(1024, 1024))


def exists_kernel_element(i,j,path='saved_models/CNTK_elements/'):
    return os.path.exists(path+f'{i}-{j}.npy')


def save_kernel(kernel,path,thread_id,reverse):
    fname = f'thread-{thread_id}.npy'
    if reverse:
        fname = f'thread-{thread_id}-rev.npy'
    np.save(path+fname,kernel)

def save_elements(elements,path,thread_id,reverse=False):
    # np.save(path+f'thread-{thread_id}.npy',elements)
    fname = f'thread-{thread_id}.p'
    if reverse:
        fname = f'thread-{thread_id}-rev.p'
    pickle.dump(elements,open(path+fname,'wb'))

def load_elements(path,thread_id):
    return pickle.load(open(path+f'thread-{thread_id}.p','rb'))

def load_H(path,thread_id):
    load_path = path+f'thread-{thread_id}.npy'
    H = np.load(load_path)
    print(f"Loaded from {load_path}, shape={H.shape}")
    return H

def CNTK_value(X,d:int,fix:bool,gap:bool,multi_thread=False,n_thread = 1,thread_id = 0,save_path='saved_models/CNTK_elements-full/',save_freq = 10000,load=False,load_kernel=False,reverse_order=False,dim=-1):
    if multi_thread:
        print(f'#threads = {n_thread}, id = {thread_id}, save_freq = {save_freq}')
    # if dim!=-1:
        # assert dim > 0
        # if not reverse_order:
            # X = X[:dim][:,:dim]

    N = X.shape[0]
    print('X.shape:',X.shape)
    n_channel = X.shape[1]
    image_shape = (X.shape[2],X.shape[3])
    n_pixel = image_shape[0]*image_shape[1]
    X = cp.asarray(X).reshape(-1, n_channel, n_pixel)
    # Calculate diagonal entries.
    L = []
    iL = []
    print("Computing Diagonal Entries")
    for i in trange(N):
        Lx, iLx = xx(X[i],d=d,fix=fix)
        L.append(Lx)
        iL.append(iLx)

    #####Calculate kernel values.
    #####Below we provide a naive implementation using for-loops.
    #####Parallelize this part according to your specific computing enviroment to utilize multiple GPUs.
    H = np.zeros((N, N), dtype=np.float32)
    elements = {}
    if load:
        elements = load_elements(save_path,thread_id)
    if load_kernel:
        H = load_H(save_path,thread_id)
        assert H.shape[0] == N and H.shape[1] == N and len(H.shape) == 2
    count = 0
    print("Computing Lower Triagnle Entries")

    range_i = np.arange(N)
    if reverse_order:
        range_i = np.flip(range_i)

    for i in tqdm(range_i):
        range_j = np.arange(i+1)
        if reverse_order:
            range_j = np.flip(range_j)
        for j in range_j:


            if multi_thread:
                if (i+j) % n_thread != thread_id:
                    continue
                # if exists_kernel_element(i,j,save_path):
                #     print(f"({i},{j}) already computed.")
                #     continue

            if load_kernel:
                if H[i][j] != 0:
                    continue

            if load:
                if i in elements:
                    if j in elements[i]:
                        H[i][j]=elements[i][j]
                        H[j][i]=H[i][j]
                        continue
            # print(f'Computing ({i},{j})')


            H[i][j] = xz(X[i], X[j], L[i], L[j], iL[i], iL[j],d=d,fix=fix,gap=gap)

            if multi_thread:
                if i not in elements:
                    elements[i] = {}
                elements[i][j] = H[i][j]
                if count % save_freq == 0:
                    save_kernel(H, save_path, thread_id, reverse_order)
                    # save_elements(elements,save_path,thread_id,reverse_order)
                    # print(f"Count = {count}, saved elements.")

            count += 1

            if j < i:
                H[j][i] = H[i][j]
    if multi_thread:
        save_kernel(H,save_path,thread_id,reverse_order)
    return H

def CNTK(X_train,X_test,d:int,fix:bool, gap:bool,
         y_train=None,y_test=None,n_class:int=None,return_kernel=True,return_pred=False,return_acc=False,
         ridge_coef=0.,train_time=np.inf,GPUs=None):
    X = np.concatenate((X_train, X_test), axis=0)
    N = X.shape[0]
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    assert len(X_train.shape) == 4
    n_channel = X_train.shape[1]
    image_shape = (X_train.shape[2],X_test.shape[3])
    n_pixel = image_shape[0]*image_shape[1]
    X = cp.asarray(X).reshape(-1, n_channel, n_pixel)

    # Calculate diagonal entries.
    L = []
    iL = []
    for i in range(N):
        Lx, iLx = xx(X[i],d=d,fix=fix)
        L.append(Lx)
        iL.append(iLx)

    #####Calculate kernel values.
    #####Below we provide a naive implementation using for-loops.
    #####Parallelize this part according to your specific computing enviroment to utilize multiple GPUs.
    H = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1):
            H[i][j] = xz(X[i], X[j], L[i], L[j], iL[i], iL[j],d=d,fix=fix,gap=gap)
            if j < i:
                H[j][i] = H[i][j]

    #####
    if return_pred or return_acc:
        # Solve kernel regression.
        Y_train = np.ones((N_train, n_class)) * (-1/n_class)
        for i in range(N_train):
            Y_train[i][y_train[i]] = 1-1/n_class

        NTK_train = H[:N_train, :N_train]
        NTK_train_ridge = NTK_train + ridge_coef*np.eye(len(NTK_train))
        NTK_test =H[N_train:, :N_train]

        time_evolution = np.eye(len(NTK_train))
        if train_time != np.inf:
            time_evolution -= scipy.linalg.expm(-train_time*NTK_train)
        u = NTK_test.dot(scipy.linalg.solve(NTK_train_ridge, time_evolution@Y_train))
        pred = np.argmax(u, axis=1)
        if return_pred:
            return pred
        if return_acc:
         return 1.0 * np.sum(pred  == y_test) / N_test

    else:
        return H


def NTK_predict(NTK_train,NTK_test,y_train,n_class,train_time=np.inf,ridge_coef=0,y_test=None):

    N_train = NTK_train.shape[0]
    N_test = NTK_test.shape[0]
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Solve kernel regression.
    Y_train = np.ones((N_train, n_class)) * (-1 / n_class)
    for i in range(N_train):
        Y_train[i][y_train[i]] = 1 - 1 / n_class

    # Y_train = np.zeros((N_train, n_class))
    # for i in range(N_train):
    #     Y_train[i][y_train[i]] = 1

    NTK_train_ridge = NTK_train + ridge_coef * np.eye(len(NTK_train))
    time_evolution = np.eye(len(NTK_train))
    if train_time != np.inf:
        time_evolution -= scipy.linalg.expm(-train_time * NTK_train)
    output = NTK_test.dot(scipy.linalg.solve(NTK_train_ridge, time_evolution @ Y_train))
    pred = np.argmax(output, axis=1)
    if y_test is None:
        return pred
    else:
        acc = 1.0 * np.sum(pred == y_test) / N_test
        return output,pred,acc


def NTK_svm_predict(NTK_train,NTK_test,y_train,y_test=None,C=1):
    N_train = NTK_train.shape[0]
    N_test = NTK_test.shape[0]
    clf = SVC(kernel = "precomputed", C = C, cache_size = 100000)
    clf.fit(NTK_train, y_train)
    pred = clf.predict(NTK_test)
    if y_test is None:
        return pred
    else:
        acc = 1.0 * np.sum(pred == y_test) / N_test
        return pred,acc

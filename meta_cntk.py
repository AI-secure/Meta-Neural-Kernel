from numpy.linalg import pinv
from numpy.linalg import matrix_power,matrix_rank,eig,eigh
from scipy.linalg import expm,pinvh,solve
from sklearn.svm import SVC,SVR
from sklearn.kernel_ridge import KernelRidge
from support.ConvNTK import *
from support.tools import *
from copy import deepcopy
from time import time
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count

class MetaCNTK():
    def __init__(self,n_class=5,print_log=True,
                 d_max=4, inner_lr=0.001, fix=False, GAP=True,
                 train_time = np.inf,
                 invMetaNTK=False,kernel_ridge=False,
                 ridge_coef=[None,None],
                 svm=False,
                 svm_coef=[1,0.1],
                 normalize_NTK:bool=False,
                 normalize_metaNTK:bool=False,
                 gpu:bool=False):
        self.n_class = n_class
        self.print_log = print_log
        self.same_X = False
        self.precomputed_NTK = False
        self.loaded_idxes = False
        self.params = {'d_max':d_max,
                       'fix':fix,
                       'GAP':GAP,
                       'inner_lr': inner_lr,
                       'train_time':train_time,
                       'invMetaNTK':invMetaNTK,
                       'kernel_ridge':kernel_ridge,
                       'ridge_coef':ridge_coef,
                       'normalize_NTK':normalize_NTK,
                       'normalize_metaNTK':normalize_metaNTK,
                       'gpu':gpu,
                       'svm':svm,
                       'svm_coef':svm_coef,
                       }
        self.old_params = deepcopy(self.params)
        self.label_transform = False
    def fit(self,X_query, Y_query, X_support, Y_support):
        self.Xs = X_query
        self.Xs_= X_support

        if len(Y_query.shape)==2:
            self.Ys = one_hot_label(Y_query,n_class=self.n_class)
            self.Ys_= one_hot_label(Y_support,n_class=self.n_class)
            self.one_hot = True
        elif len(Y_query.shape)==3:
            self.Ys = Y_query
            self.Ys_ = Y_support
            self.one_hot = False
        else:
            raise ValueError()
        self.im_shape = X_query.shape[-3:]
        assert len(X_query) == len(Y_query) == len(Y_support) == len(X_support)
        self.N_train = len(X_query) # number of training tasks
        self.new_dataset = True

    def update_params(self,**params):
        self.old_params = deepcopy(self.params)
        for key,value in params.items():
            assert isinstance(value,int) or isinstance(value,float) or isinstance(value, bool) or isinstance(value,list) or isinstance(value,np.ndarray)
            assert key in self.params
            self.params[key] = value



    def load_precompute_NTKs(self,NTK_all):
        self.precomputed_NTK = True
        self.ntk_all = NTK_all
    def load_idxes(self,idx_Xs,idx_Xs_,idx_test_Xs,idx_test_Xs_):
        self.idx_Xs = idx_Xs
        self.idx_Xs_ = idx_Xs_
        self.idx_test_Xs = idx_test_Xs
        self.idx_test_Xs_ = idx_test_Xs_
        self.loaded_idxes = True
    def check_if_need_to_recalculate_NTKs(self):
        self.recal_NTK,self.recal_T,self.recal_MetaNTK= True,True,True # if recalculate NTK/T/MetaNTK
        if self.new_dataset:
            # If just load a new dataset, we have to recalculate all values
            self.new_dataset = False
            return
        # Check if NTKs and MetaNTKs need to be re-calculated
        NTK_keys = ['d_max','fix','GAP','normalize_NTK']
        T_Keys = ['kernel_ridge','ridge_coef','inner_lr']
        MetaNTK_keys = ['normalize_metaNTK']
        if check_if_values_unchanged(self.old_params,self.params,NTK_keys) == True:
            self.recal_NTK = False
        if check_if_values_unchanged(self.old_params,self.params,MetaNTK_keys) == True:
            self.recal_MetaNTK = False
        if check_if_values_unchanged(self.old_params,self.params,T_Keys) == True:
            self.recal_T = False

        if self.recal_NTK:
            self.recal_T,self.recal_MetaNTK = True,True
        if self.recal_T:
            self.recal_MetaNTK = True

        if self.print_log:
            if self.recal_NTK or self.recal_T or self.recal_MetaNTK:
                to_compute = []
                if self.recal_NTK: to_compute.append('NTK')
                if self.recal_T: to_compute.append('T')
                if self.recal_MetaNTK: to_compute.append('MetaNTK')
                print("Need to compute:",to_compute)

    def load_test_tasks(self,X_query, X_support, Y_support):
        self.test_Xs = X_query
        self.test_Xs_ = X_support
        # if self.params['svm']:
        #     self.test_Ys_ = Y_support
        # else:
        if len(Y_support.shape)==2:
            self.test_Ys_ = one_hot_label(Y_support,n_class=self.n_class)
        elif len(Y_support.shape)==3:
            self.test_Ys_ = Y_support
        else:
            raise ValueError()
        self.N_test = len(X_query) # number of test tasks
        self.same_X = False
        if self.Xs.shape == self.test_Xs.shape and self.Xs_.shape == self.test_Xs_.shape:
            if np.allclose(self.Xs,self.test_Xs) and np.allclose(self.Xs_,self.test_Xs_):
                self.same_X = True


    def update_labels(self,Y_qry=None,Y_spt=None,test_Y_qry=None,test_Y_spt=None):
        if Y_qry is not None:
            self.Ys = Y_qry
        if Y_spt is not None:
            self.Ys_ = Y_spt
        if test_Y_qry is not None:
            self.test_Ys = test_Y_qry
        if test_Y_spt is not None:
            self.test_Ys_ = test_Y_spt

    def cal_NTKs(self,ridge=False):
        if self.recal_NTK:
            self.Xs_all, idxes = reshape_cat([self.Xs, self.Xs_, self.test_Xs, self.test_Xs_],
                                             shape=(-1, *self.im_shape), return_idxes=True)

            if not self.loaded_idxes:
                self.idx_Xs, self.idx_Xs_, self.idx_test_Xs, self.idx_test_Xs_ = idxes

            self.all_idxes = [[self.idx_test_Xs_, self.idx_test_Xs], [self.idx_Xs_, self.idx_Xs]]

            if not self.precomputed_NTK:
                if self.print_log:
                    print("Calculating CNTK...")
                t0 = time()
                self.ntk_all = CNTK_value(self.Xs_all,d=self.params['d_max'],fix=self.params['fix'],gap=self.params['GAP'])
                if self.print_log:
                    print(f" Took {round(time()-t0,2)}s.")


            if ridge:
                self.ntk_all_ridge,ridge_coef = ridge_reg(self.ntk_all,self.params['ridge_coef'][0],return_coef=True)
                if self.print_log and ridge_coef > 0:
                    print("NTK_all ridge coef = %.3g"%ridge_coef)
                if self.params['normalize_NTK']:
                    self.ntk_all = normalize_kernel(self.ntk_all_ridge)
                    self.ntk_all_ridge = self.ntk_all
                    # self.ntk_all_ridge,ridge_coef = ridge_reg(self.ntk_all,self.params['ridge_coef'][0],return_coef=True)


            else:
                self.ntk_all_ridge = self.ntk_all




    def cal_Ts(self,ridge=True,check_rank=False):
        if self.recal_T:
            if self.print_log:
                print("Calculating T values.")
            self.T_Xs_ = []
            self.T_test_Xs_ = []
            for train in [True,False]:
                N = len(self.Xs_) if train else len(self.test_Xs_)
                for idx in range(N):
                    ntk = self.get_ntk(fst_train=train,fst_idx=idx,fst_qry=False,
                                       snd_train=train,snd_idx=idx,snd_qry=False)
                    ntk_ridge = self.get_ntk(fst_train=train,fst_idx=idx,fst_qry=False,
                                       snd_train=train,snd_idx=idx,snd_qry=False,ridge=True)

                    inv_ntk = pinv(ntk_ridge)

                    T = self.T(ntk=ntk,inv_ntk=inv_ntk)
                    if train:
                        self.T_Xs_.append(T)
                    else:
                        self.T_test_Xs_.append(T)

            self.all_Ts = [self.T_Xs_,self.T_test_Xs_]

    def cal_metaNTKs(self,ridge=True):
        if self.recal_MetaNTK:
            if self.print_log:
                print("Calculating MetaNTK values.")
            self.metaNTK_train = self.metaNTK(fst_test=False,snd_test=False) # MetaNTK of Train <- Train
            self.metaNTK_test  = self.metaNTK(fst_test=True,snd_test=False) # MetaNTK of Test <- Train
            if self.params['normalize_metaNTK']:
                N_train = len(self.metaNTK_train)
                N_test = len(self.metaNTK_test)
                N = N_train + N_test
                self.metaNTK_all = np.zeros((N,N))
                self.metaNTK_all[:N_train,:N_train] = self.metaNTK_train
                self.metaNTK_all[N_train:,:N_train] = self.metaNTK_test
                self.metaNTK_all[:N_train,N_train:] = self.metaNTK_test.T
                self.metaNTK_all[N_train:,N_train:] = self.metaNTK(fst_test=True,snd_test=True)
                assert np.allclose(self.metaNTK_all, self.metaNTK_all.T, rtol=1e-5, atol=1e-5)
                if ridge:
                    self.metaNTK_all_ridge,train_ridge_coef = ridge_reg(self.metaNTK_all,self.params['ridge_coef'][1], return_coef=True)
                    if self.print_log and train_ridge_coef > 0:
                        print("MetaNTK_train ridge coef = %.3g" % train_ridge_coef)
                self.normalized_metaNTK_all = normalize_kernel(self.metaNTK_all_ridge)
                self.metaNTK_train = self.normalized_metaNTK_all[:N_train,:N_train]
                self.metaNTK_test  = self.normalized_metaNTK_all[N_train:,:N_train]
                self.metaNTK_train_ridge = self.metaNTK_train

            else:
                if ridge:
                    self.metaNTK_train_ridge,train_ridge_coef = ridge_reg(self.metaNTK_train,self.params['ridge_coef'][1], return_coef=True)
                    if self.print_log and train_ridge_coef > 0:
                        print("MetaNTK_train ridge coef = %.3g" % train_ridge_coef)

    def get_T(self,for_test:bool,idx:int):
        return self.all_Ts[for_test][idx]

    def get_ntk(self,fst_train:bool,fst_idx:int,fst_qry:bool,
                    snd_train:bool,snd_idx:int,snd_qry:bool,ridge=False):
        self.idx_1 = self.all_idxes[fst_train][fst_qry][fst_idx]
        self.idx_2 = self.all_idxes[snd_train][snd_qry][snd_idx]
        if ridge:
            return self.ntk_all_ridge[self.idx_1][:,self.idx_2]
        else:
            return self.ntk_all[self.idx_1][:,self.idx_2]
        
    def get_SVM_NTK(self,for_test:bool):
        if self.params['kernel_ridge']:
            clf = KernelRidge(alpha=self.params['ridge_coef'][0], kernel="precomputed")
        else:
            clf = SVR(kernel = "precomputed", C=self.params['svm_coef'][0], epsilon=self.params['svm_coef'][1], cache_size = 100000)
        output = []
        train = not for_test
        Ys_ = self.test_Ys_ if for_test else self.Ys_
        N = self.N_test if for_test else self.N_train
        for idx in range(N):
            NTK_train = self.get_ntk(fst_train=train, fst_idx=idx,fst_qry=False, snd_train=train, snd_idx=idx,snd_qry=False,ridge=True)
            NTK_test  = self.get_ntk(fst_train=train, fst_idx=idx,fst_qry=True, snd_train=train, snd_idx=idx,snd_qry=False,ridge=False)
            y   = Ys_[idx]
            time_evolution = self.time_evolution(NTK_train,self.params['inner_lr'])
            clf.fit(X=NTK_train,y=time_evolution@y)
            pred = clf.predict(X=NTK_test)
            output.append(pred)
        return np.concatenate(output)

    def get_SVM_metaNTK(self, eff_Ys): # effective Ys
        if self.params['kernel_ridge']:
            clf = KernelRidge(alpha=self.params['ridge_coef'][1], kernel="precomputed")
        else:
            clf = SVR(kernel = "precomputed",  C=self.params['svm_coef'][0],epsilon=self.params['svm_coef'][1], cache_size = 1000000)
        clf.fit(X=self.metaNTK_train_ridge,y=eff_Ys)
        return clf.predict(self.metaNTK_test)

    def predict(self,X_query=None,X_support=None,Y_support=None):
        # self.print_log = print_log
        if X_query is None: X_query = self.test_Xs
        else: self.test_Xs = X_query
        if X_support is None: X_support = self.test_Xs_
        else: self.test_Xs_ = X_support
        if Y_support is None: Y_support = self.test_Ys_
        else: self.test_Ys_ = Y_support
        assert len(X_query) == len(Y_support) == len(X_support)
        self.N_test = len(X_query)
        self.check_if_need_to_recalculate_NTKs()
        self.cal_NTKs(ridge=True)
        self.cal_Ts(ridge=False)
        # if not self.same_X:
        self.cal_metaNTKs(ridge=True)

        Ys = np.concatenate(self.Ys, axis=0)
        if self.params['kernel_ridge'] or self.params['svm']:
            svm_NTK_test = self.get_SVM_NTK(for_test=True)
            svm_NTK_train = self.get_SVM_NTK(for_test=False)
            time_evolution = self.time_evolution(self.metaNTK_train, train_time=self.params['train_time'])
            self.eff_Ys = time_evolution@(Ys-svm_NTK_train)
            svm_metaNTK = self.get_SVM_metaNTK(eff_Ys=self.eff_Ys)
            self.term13 = svm_metaNTK
            self.term2 = svm_NTK_test
            output = self.term13 + self.term2

        else:
            self.test_NTK_T_Ys_ = self.get_NTK_T_Ys_(for_test=True)
            self.train_NTK_T_Ys_= self.get_NTK_T_Ys_(for_test=False)
            self.term2 = self.test_NTK_T_Ys_


            if self.params['invMetaNTK']:
                self.inv_metaNTK_train = pinvh(self.metaNTK_train_ridge)
                meta_T = self.T(self.metaNTK_train, self.inv_metaNTK_train, self.params['train_time'])
                self.meta_inv_meta = self.metaNTK_test@self.inv_metaNTK_train@meta_T
                self.term13 = self.meta_inv_meta@(Ys-self.train_NTK_T_Ys_)
            else:
                time_evolution = self.time_evolution(self.metaNTK_train,train_time=self.params['train_time'])
                self.term13 = self.metaNTK_test@solve(self.metaNTK_train_ridge,time_evolution@(Ys-self.train_NTK_T_Ys_))

            output = self.term13+self.term2
        if self.one_hot:
            pred = pred_from_one_hot(output)
            self.pred = pred
            return pred.reshape(*self.test_Xs.shape[:2])
        else:
            pred = output
            self.pred = pred
            return pred.reshape(*self.test_Xs.shape[:2],-1)



    def get_NTK_T_Ys_(self,for_test:bool):
        output = []
        train = not for_test
        if not self.label_transform:
            Ys_ = self.test_Ys_ if for_test else self.Ys_
        else:
            Ys_ = self.label_test_Ys_ if for_test else self.label_Ys_
            transMat = self.test_transMat if for_test else self.transMat
        N = self.N_test if for_test else self.N_train
        for idx in range(N):
            ntk_qry = self.get_ntk(fst_train=train, fst_idx=idx,fst_qry=True, snd_train=train, snd_idx=idx,snd_qry=False)
            ntk_spt = self.get_ntk(fst_train=train, fst_idx=idx, fst_qry=False,
                                     snd_train=train, snd_idx=idx, snd_qry=False, ridge=False)
            ntk_spt_ridge = self.get_ntk(fst_train=train, fst_idx=idx, fst_qry=False,
                                     snd_train=train, snd_idx=idx, snd_qry=False, ridge=True)
            # T   = self.get_T(for_test=for_test,idx=idx)
            y   = Ys_[idx]
            # ntk_T_y_ = ntk_qry@T@y
            time_evolution = self.time_evolution(ntk_spt,self.params['inner_lr'])
            y = time_evolution@y
            ntk_T_y_ = ntk_qry@solve(ntk_spt_ridge,y)
            if self.label_transform:
                self.ntk_T_y_ = ntk_T_y_
                ntk_T_y_ = ntk_T_y_@transMat[idx]
            output.append(ntk_T_y_)
        return np.concatenate(output)

    def single_metaNTK(self,fst_test:bool,idx_1:int,snd_test:bool,idx_2:int):

        # idx_1 is the task idx for the first argument (test/train), while idx_2 is for the second (train).
        T1 = self.get_T(for_test=fst_test,idx=idx_1)
        T2 = self.get_T(for_test=snd_test,idx=idx_2)
        T2_t = T2.transpose()
        fst_train= not fst_test
        snd_train = not snd_test
        ntk12 = self.get_ntk(fst_train=fst_train,fst_idx=idx_1,fst_qry=True,snd_train=snd_train,snd_idx=idx_2,snd_qry=True)
        ntk11_= self.get_ntk(fst_train=fst_train,fst_idx=idx_1,fst_qry=True,snd_train=fst_train,snd_idx=idx_1,snd_qry=False)
        ntk1_2_=self.get_ntk(fst_train=fst_train,fst_idx=idx_1,fst_qry=False,snd_train=snd_train,snd_idx=idx_2,snd_qry=False)
        ntk2_2 =self.get_ntk(fst_train=snd_train,fst_idx=idx_2,fst_qry=False,snd_train=snd_train, snd_idx=idx_2, snd_qry=True)
        ntk1_2=self.get_ntk(fst_train=fst_train,fst_idx=idx_1,fst_qry=False,snd_train=snd_train,snd_idx=idx_2,snd_qry=True)
        ntk12_=self.get_ntk(fst_train=fst_train,fst_idx=idx_1,fst_qry=True,snd_train=snd_train,snd_idx=idx_2,snd_qry=False)
        # Terms w/ Order in T
        order_0 = ntk12
        order_1 = - ntk11_@T1@ntk1_2 - ntk12_@T2_t@ntk2_2
        order_2 = ntk11_@T1@ntk1_2_@T2_t@ntk2_2
        return order_0 + order_1+order_2
    def metaNTK(self,fst_test:bool,snd_test:bool):
        N_1 = self.N_test if fst_test else self.N_train
        N_2 = self.N_test if snd_test else self.N_train
        shape_0 = self.test_Xs.shape[1] if fst_test else self.Xs.shape[1]
        shape_1 = self.test_Xs.shape[1] if snd_test else self.Xs.shape[1]
        # shape_1 = self.Xs.shape[1]
        placeholder = np.zeros((shape_0,shape_1))
        meta_ntks = [[placeholder for _ in range(N_2)] for _ in range(N_1)]
        for idx_1 in range(N_1):
            # If for_train, then we only need to fill in the lower triangle,
            # and take a transpose of it is the upper one.
            N = N_2 if fst_test != snd_test else idx_1 + 1

            for idx_2 in range(N):

                meta_ntk = self.single_metaNTK(fst_test=fst_test, idx_1=idx_1, snd_test=snd_test,idx_2=idx_2)
                meta_ntks[idx_1][idx_2] = meta_ntk
                assert meta_ntk.shape == placeholder.shape
                if fst_test == snd_test:
                    meta_ntks[idx_2][idx_1]=np.transpose(meta_ntk)


        meta_ntks = np.array(meta_ntks)
        metaNTK = np.hstack(np.hstack(meta_ntks))

        return metaNTK#, all_ntks, all_inv_ntks


    # This function will be different for multi-dim input or image
    def flatten(self,X,is_input=True):
        return X.reshape(-1,1)
    def reshape(self,X,shape):
        if isinstance(X,list):
            output = []
            for x in X:
                output.append(x.reshape(*shape))
            return tuple(output)
        else:
            return X.reshape(*shape)


    def invNTKs_single_tasks(self, ntks,var_names:list=['test_X_','X_']):
        inv_ntks = {}
        for var_name in var_names:
            inv_ntks[var_name] = ntks[var_name][var_name]
        return inv_ntks

    def time_evolution(self,ntk,train_time):
        I = np.eye(len(ntk))
        if train_time == np.inf:
            return I
        else:
            return I - expm(-train_time*ntk)

    def T(self,ntk, inv_ntk=None, inner_lr=None):
        if inv_ntk is None:
            inv_ntk = pinvh(ntk)
        if inner_lr is None:
            inner_lr = self.params['inner_lr']

        if inner_lr == np.inf:
            return inv_ntk
        else:
            return inv_ntk @ self.time_evolution(ntk,inner_lr)

    def load_label_and_transform_mat(self,Y_spt,test_Y_spt,transMat,test_transMat):
        self.label_transform=True
        self.label_Ys_ = one_hot_label(Y_spt,n_class=self.n_class)
        self.label_test_Ys_= one_hot_label(test_Y_spt,n_class=self.n_class)
        self.transMat = transMat
        self.test_transMat = test_transMat

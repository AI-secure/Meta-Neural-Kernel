import argparse
import time
import typing

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
import higher

import pickle
from tqdm.notebook import tqdm,trange
from support.omniglot_loaders import OmniglotNShot
from IPython.display import clear_output
def torch2np(tensors):
    output = []
    for tensor in tensors:
        if isinstance(tensor,torch.Tensor):
            output.append(tensor.data.cpu().numpy())
        else: output.append(tensor)
    return output
def np2torch(arrays,device=torch.device('cuda'), label_long_type = True):
    outputs = []
    for array in arrays:
        if isinstance(array,np.ndarray):
            output = torch.from_numpy(array).float().to(device)
        else:
            output = array
        if len(output.shape) < 5:
            if label_long_type:
                output = output.long()
        outputs.append(output)
    return outputs

def remove_padding(arrays):
    # Original Images in Omniglot are of 28*28.
    # However, a CUDA package we to calculate Meta Neural Kernels requires 32*32 images.
    # Hence, we add paddings to 28*28 images to enlarge them to 32*32 for Meta Neural Kernels computing.
    # For normal MAML, we can just remove the paddings.
    if not isinstance(arrays,list):
        return arrays[:,:,:,2:-2,2:-2]
    outputs = []
    for array in arrays:
        if array.shape[-1] == 32 and array.shape[-2] == 32:
            output = array[:,:,:,2:-2,2:-2]
            outputs.append(output)
    return outputs

def cross_entropy_acc_fn(preds,labels):
    return
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

class PredictionFromEmbeddings():
    def __init__(self,dataset,meta_test:bool,n_neighbors=1):
        self.dataset = dataset
        self.preds = []
        self.test_accs = []
        self.n_neighbors = n_neighbors
        self.all_embeddings = []
        self.all_labels = []
        self.meta_test = meta_test

    def loss(self):
        pred_embeddings = np.array(self.all_embeddings)
        label_embeddings = np.array(self.dataset.test_Y_qry_emb)
        loss = np.mean( (pred_embeddings-label_embeddings)**2)
        return loss
    def pred_from_emb(self,embeddings,task_idx):


        embeddings = embeddings.detach().data.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
        if self.meta_test:
            centroids = self.dataset.test_Y_centroids[task_idx]
            labels = self.dataset.test_Y_qry[task_idx]
        else:
            centroids = self.dataset.Y_centroids[task_idx]
            labels = self.dataset.Y_qry[task_idx]
        nbrs.fit(centroids[task_idx])
        _, pred = nbrs.kneighbors(embeddings)
        pred = pred.flatten()

        assert pred.shape == labels.shape
        acc = np.mean(pred == labels)
        self.all_embeddings.append(embeddings)
        self.preds.append(pred)
        self.all_labels.append(labels)
        self.test_accs.append(acc)
        return acc

    def __call__(self, embeddings, task_idx):
        return self.pred_from_emb(embeddings=embeddings,task_idx=task_idx)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def build_MAML_model(n_way,device,lr=1e-3,n_channel=64,batch_norm = True):
    if batch_norm:
        net = nn.Sequential(
            nn.Conv2d(1, n_channel, 3),
            nn.BatchNorm2d(n_channel, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_channel, n_channel, 3),
            nn.BatchNorm2d(n_channel, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_channel, n_channel, 3),
            nn.BatchNorm2d(n_channel, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(n_channel, n_way)).to(device)
    else:
        net = nn.Sequential(
            nn.Conv2d(1, n_channel, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_channel, n_channel, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_channel, n_channel, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(n_channel, n_way)).to(device)

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_opt = optim.Adam(net.parameters(), lr=lr)
    return net,meta_opt


def MAML_train_fn(tasks, net, device, meta_opt, epoch, log, batch_size=100, l2_loss=False,dataset=None,n_neighbors=1):
    net.train()
    n_train_iter = 1
    X_spt, Y_spt, X_qry, Y_qry = tasks
    n_task = len(X_spt)
    task_batch_size = min(batch_size,n_task)

    loss_fn = F.mse_loss if l2_loss else F.cross_entropy
    if l2_loss:
        # Define the Prediction Transformer that can transform predicted embeddings into class labels
        pred_transformer = PredictionFromEmbeddings(dataset=dataset,n_neighbors=n_neighbors)

    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        task_idxes = np.random.choice(n_task,size=task_batch_size)
        x_spt, y_spt, x_qry, y_qry = X_spt[task_idxes], Y_spt[task_idxes], X_qry[task_idxes], Y_qry[task_idxes]
        task_num, setsz, c_, h, w = x_spt.size()

        querysz = x_qry.size(1)


        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()
        for i in range(task_num):
            with higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = loss_fn(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_output = fnet(x_qry[i])
                qry_loss = loss_fn(qry_output, y_qry[i])
                qry_losses.append(qry_loss.detach().mean().item())
                if l2_loss:
                    qry_acc = pred_transformer.pred_from_emb(qry_output, i)
                    qry_accs.append(qry_acc)
                else:
                    qry_accs.append(
                        (qry_output.argmax(dim=1) == y_qry[i]).detach())

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward()
        qry_losses = np.mean(qry_losses)
        if l2_loss:
            qry_accs = 100 * np.mean(qry_accs)
        else:
            qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
        meta_opt.step()
        qry_losses = np.mean(qry_losses)
        # qry_accs = 100. * sum(qry_accs) / task_num
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time
#         if batch_idx % 4 == 0:
#             print(
#                 f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
#             )

        log.append({
            'epoch': epoch+1,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'train',
            'time': time.time(),
        })

def MAML_test_fn(tasks, net, device, epoch, log, l2_loss=False,dataset=None,n_neighbors=1):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    net.train()
    n_test_iter = 1

    qry_losses = []
    qry_accs = []
    loss_fn = F.mse_loss if l2_loss else F.cross_entropy
    if l2_loss:
        # Define the Prediction Transformer that can transform predicted embeddings into class labels
        pred_transformer = PredictionFromEmbeddings(dataset=dataset,n_neighbors=n_neighbors)

    x_spt, y_spt, x_qry, y_qry = tasks


    task_num, setsz, c_, h, w = x_spt.size()
    querysz = x_qry.size(1)

    # TODO: Maybe pull this out into a separate module so it
    # doesn't have to be duplicated between `train` and `test`?
    n_inner_iter = 5
    inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

    for i in range(task_num):
        with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.
            for _ in range(n_inner_iter):
                spt_logits = fnet(x_spt[i])
                spt_loss = loss_fn(spt_logits, y_spt[i])
                diffopt.step(spt_loss)

            # The query loss and acc induced by these parameters.
            qry_output = fnet(x_qry[i]).detach()
            qry_loss = loss_fn(
                qry_output, y_qry[i], reduction='none')
            qry_losses.append(qry_loss.detach().mean().item())
            if l2_loss:
                qry_acc=pred_transformer.pred_from_emb(qry_output,i)
                qry_accs.append(qry_acc)
            else:
                qry_accs.append(
                    (qry_output.argmax(dim=1) == y_qry[i]).detach())

    qry_losses = np.mean(qry_losses)
    if l2_loss:
        qry_accs = 100*np.mean(qry_accs)
    else:
        qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
#     print(
#         f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}'
#     )
    log.append({
        'epoch': epoch + 1,
        'loss': qry_losses,
        'acc': qry_accs,
        'mode': 'test',
        'time': time.time(),
    })


    if l2_loss:
        return qry_accs,pred_transformer
    else:
        return qry_accs





def plot(log):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    plt.show()


def train_MAML(db, net, device, meta_opt, epoch, log,verbose=True,l2_loss=False):
    if l2_loss:
        loss_fn = F.mse_loss
    else:
        loss_fn = F.cross_entropy

    net.train()
    n_train_iter = db.x_train.shape[0] // db.batchsz
    assert n_train_iter >= 1
    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db.next()

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)


        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()
        for i in range(task_num):
            with higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = loss_fn(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_logits = fnet(x_qry[i])
                qry_loss = loss_fn(qry_logits, y_qry[i])
                qry_losses.append(qry_loss.detach())
                if not l2_loss:
                    qry_acc = (qry_logits.argmax(
                        dim=1) == y_qry[i]).sum().item() / querysz
                    qry_accs.append(qry_acc)

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward()

        meta_opt.step()
        qry_losses = sum(qry_losses) / task_num
        if not l2_loss:
            qry_accs = 100. * sum(qry_accs) / task_num

        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time
        # if batch_idx % 4 == 0:
        #     if verbose:
        #         print(f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}')

        log.append({
            'epoch': i,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'train',
            'time': time.time(),
        })
    return qry_accs

def test_MAML(db, net, device, epoch, log, test_tasks=None,verbose=True,l2_loss=False,dataset=None):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    net.train()
    if l2_loss:
        loss_fn = F.mse_loss
    else:
        loss_fn = F.cross_entropy

    qry_losses = []
    qry_accs = []
    if test_tasks is not None:
        n_test_iter = 1
    else:
        n_test_iter = db.x_test.shape[0] // db.batchsz

    pred_embs = []
    for batch_idx in range(n_test_iter):
        if test_tasks is not None:
            x_spt, y_spt, x_qry, y_qry = test_tasks
            [x_spt, y_spt, x_qry, y_qry] = np2torch([x_spt, y_spt, x_qry, y_qry],device=device,
                                                    label_long_type=not l2_loss)
            if not l2_loss:
                y_qry = y_qry.long()
                y_spt = y_spt.long()
            if verbose:
                print('--- Read given test tasks.')
        else:
            x_spt, y_spt, x_qry, y_qry = db.next('test')


        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)


        n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)


        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = loss_fn(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The query loss and acc induced by these parameters.
                qry_logits = fnet(x_qry[i]).detach()
                qry_loss = loss_fn(
                    qry_logits, y_qry[i], reduction='none')
                qry_losses.append(qry_loss.detach())
                if not l2_loss:
                    qry_accs.append((qry_logits.argmax(dim=1) == y_qry[i]).detach())
                else:
                    pred_embs.append(qry_logits.cpu().numpy())


    qry_losses = torch.cat(qry_losses).mean().item()
    if not l2_loss:
        qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
    else:
        pred_embs = np.array(pred_embs)
        pred_test_Y =pred_from_emb(pred_embs,dataset)
        pred_test_Y = pred_test_Y.reshape(*dataset.test_Y_qry.shape)
        test_acc = np.mean(pred_test_Y == dataset.test_Y_qry)
        qry_accs = 100.*test_acc

    if verbose:
        print(f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}')
    log.append({
        'epoch': epoch + 1,
        'loss': qry_losses,
        'acc': qry_accs,
        'mode': 'test',
        'time': time.time(),
    })


    return qry_accs

def database_for_maml(dataset):

    tasks = vars(dataset)

    X = np.concatenate([tasks['X_qry'],tasks['X_spt']],axis=1)
    Y = np.concatenate([tasks['Y_qry'],tasks['Y_spt']],axis=1)
    new_X =[]
    n_way = 5
    for x,y in zip(X,Y):
        idxes = np.argsort(y).reshape(n_way,-1)
        for i in range(idxes.shape[0]):
            new_X.append([])
            for j in range(idxes.shape[1]):
                idx = idxes[i,j]
                new_X[-1].append(x[idx])
    x_train = remove_padding(np.array(new_X))

    test_tasks = remove_padding(tasks['test_X_spt']), tasks['test_Y_spt'],remove_padding(tasks['test_X_qry']), tasks['test_Y_qry']
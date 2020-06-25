"""
Meta-learning Omniglot and mini-imagenet experiments with iMAML-GD (see [1] for more details).

The code is quite simple and easy to read thanks to the following two libraries which need both to be installed.
- higher: https://github.com/facebookresearch/higher (used to get stateless version of torch nn.Module-s)
- torchmeta: https://github.com/tristandeleu/pytorch-meta (used for meta-dataset loading and minibatching)


[1] Rajeswaran, A., Finn, C., Kakade, S. M., & Levine, S. (2019).
    Meta-learning with implicit gradients. In Advances in Neural Information Processing Systems (pp. 113-124).
    https://arxiv.org/abs/1909.04630
"""
import math
import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

import higher
from maml_utils import np2torch
import hypergrad as hg
import tqdm
class Task:
    """
    Handles the train and valdation loss for a single task
    """
    def __init__(self, reg_param, meta_model, data, batch_size=None):
        device = next(meta_model.parameters()).device

        # stateless version of meta_model
        self.fmodel = higher.monkeypatch(meta_model, device=device, copy_initial_weights=True)

        self.n_params = len(list(meta_model.parameters()))
        self.train_input, self.train_target, self.test_input, self.test_target = data
        self.reg_param = reg_param
        self.batch_size = 1 if not batch_size else batch_size
        self.val_loss, self.val_acc = None, None

    def bias_reg_f(self, bias, params):
        # l2 biased regularization
        return sum([((b - p) ** 2).sum() for b, p in zip(bias, params)])

    def train_loss_f(self, params, hparams):
        # biased regularized cross-entropy loss where the bias are the meta-parameters in hparams
        out = self.fmodel(self.train_input, params=params)
        return F.cross_entropy(out, self.train_target) + 0.5 * self.reg_param * self.bias_reg_f(hparams, params)

    def val_loss_f(self, params, hparams):
        # cross-entropy loss (uses only the task-specific weights in params
        out = self.fmodel(self.test_input, params=params)
        val_loss = F.cross_entropy(out, self.test_target)/self.batch_size
        self.val_loss = val_loss.item()  # avoid memory leaks

        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        self.val_acc = pred.eq(self.test_target.view_as(pred)).sum().item() / len(self.test_target)

        return val_loss




def inner_loop(hparams, params, optim, n_steps, log_interval, create_graph=False):
    params_history = [optim.get_opt_params(params)]

    for t in range(n_steps):
        params_history.append(optim(params_history[-1], hparams, create_graph=create_graph))

        if log_interval and (t % log_interval == 0 or t == n_steps-1):
            print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))

    return params_history


def test_imaml(test_tasks, meta_model, n_steps, get_inner_opt, reg_param, log_interval=None):
    meta_model.train()
    device = next(meta_model.parameters()).device

    val_losses, val_accs = [], []
#     for k, batch in enumerate(dataloader):
    tr_xs,tr_ys,tst_xs,tst_ys = test_tasks
#         tr_xs, tr_ys = batch["train"][0].to(device), batch["train"][1].to(device)
#         tst_xs, tst_ys = batch["test"][0].to(device), batch["test"][1].to(device)
    tr_xs, tr_ys, tst_xs, tst_ys = np2torch([tr_xs, tr_ys, tst_xs, tst_ys],device=device,label_long_type=True)
    for t_idx, (tr_x, tr_y, tst_x, tst_y) in enumerate(zip(tr_xs, tr_ys, tst_xs, tst_ys)):

        task = Task(reg_param, meta_model, (tr_x, tr_y, tst_x, tst_y))
        inner_opt = get_inner_opt(task.train_loss_f)

        params = [p.detach().clone().requires_grad_(True) for p in meta_model.parameters()]
        last_param = inner_loop(meta_model.parameters(), params, inner_opt, n_steps, log_interval=log_interval)[-1]

        task.val_loss_f(last_param, meta_model.parameters())

        val_losses.append(task.val_loss)
        val_accs.append(task.val_acc)

#         if len(val_accs) >= n_tasks:
    return np.array(val_losses), np.array(val_accs)


def get_cnn_omniglot(hidden_size, n_classes):
    def conv_layer(ic, oc, ):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=True # When this is true is called the "transductive setting"
                           )
        )

    net =  nn.Sequential(
        conv_layer(1, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten(),
        nn.Linear(hidden_size, n_classes)
    )

    initialize(net)
    return net


def initialize(net):
    # initialize weights properly
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            #m.weight.data.normal_(0, 0.01)
            #m.bias.data = torch.ones(m.bias.data.size())
            m.weight.data.zero_()
            m.bias.data.zero_()

    return net



def get_inner_opt(train_loss):
    inner_opt_class = hg.GradientDescent
    inner_opt_kwargs = {'step_size': .1}
    return inner_opt_class(train_loss, **inner_opt_kwargs)

def train_imaml(meta_model,db,reg_param,hg_mode,K,T,outer_opt,inner_log_interval,notebook=True):
    meta_model.train()
    n_train_iter = db.x_train.shape[0] // db.batchsz
    for batch_idx in range(n_train_iter):
        #     tr_xs, tr_ys = batch["train"][0].to(device), batch["train"][1].to(device)
        #     tst_xs, tst_ys = batch["test"][0].to(device), batch["test"][1].to(device)
        tr_xs, tr_ys, tst_xs, tst_ys = db.next()
        outer_opt.zero_grad()

        val_loss, val_acc = 0, 0
        forward_time, backward_time = 0, 0
        for t_idx, (tr_x, tr_y, tst_x, tst_y) in enumerate(zip(tr_xs, tr_ys, tst_xs, tst_ys)):
            start_time_task = time.time()

            # single task set up
            task = Task(reg_param, meta_model, (tr_x, tr_y, tst_x, tst_y), batch_size=tr_xs.shape[0])
            inner_opt = get_inner_opt(task.train_loss_f)

            # single task inner loop
            params = [p.detach().clone().requires_grad_(True) for p in meta_model.parameters()]
            last_param = inner_loop(meta_model.parameters(), params, inner_opt, T, log_interval=inner_log_interval)[-1]
            forward_time_task = time.time() - start_time_task

            # single task hypergradient computation
            if hg_mode == 'CG':
                # This is the approximation used in the paper CG stands for conjugate gradient
                cg_fp_map = hg.GradientDescent(loss_f=task.train_loss_f, step_size=1.)
                hg.CG(last_param, list(meta_model.parameters()), K=K, fp_map=cg_fp_map, outer_loss=task.val_loss_f)
            elif hg_mode == 'fixed_point':
                hg.fixed_point(last_param, list(meta_model.parameters()), K=K, fp_map=inner_opt,
                               outer_loss=task.val_loss_f)

            backward_time_task = time.time() - start_time_task - forward_time_task

            val_loss += task.val_loss
            val_acc += task.val_acc / task.batch_size

            forward_time += forward_time_task
            backward_time += backward_time_task

    outer_opt.step()



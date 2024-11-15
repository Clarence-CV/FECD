import os
import torch
import inspect

from datetime import datetime
from torch.utils.data.sampler import Sampler
from loguru import logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def init_experiment(args, runner_name=None, exp_id=None):
#     # Get filepath of calling script
#     if runner_name is None:
#         runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]
#
#     root_dir = os.path.join(args.exp_root, *runner_name)
#
#     if not os.path.exists(root_dir):
#         os.makedirs(root_dir)
#
#     # Either generate a unique experiment ID, or use one which is passed
#     if exp_id is None:
#
#         if args.exp_name is None:
#             raise ValueError("Need to specify the experiment name")
#         # Unique identifier for experiment
#         now = '{}_({:02d}.{:02d}.{}_|_'.format(args.exp_name, datetime.now().day, datetime.now().month,
#                                                datetime.now().year) + \
#               datetime.now().strftime("%S.%f")[:-3] + ')'
#
#         log_dir = os.path.join(root_dir, 'log', now)
#         while os.path.exists(log_dir):
#             now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
#                   datetime.now().strftime("%S.%f")[:-3] + ')'
#
#             log_dir = os.path.join(root_dir, 'log', now)
#
#     else:
#
#         log_dir = os.path.join(root_dir, 'log', f'{exp_id}')
#
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#
#     logger.add(os.path.join(log_dir, 'log.txt'))
#     args.logger = logger
#     args.log_dir = log_dir
#
#     # Instantiate directory to save models to
#     model_root_dir = os.path.join(args.log_dir, 'checkpoints')
#     if not os.path.exists(model_root_dir):
#         os.mkdir(model_root_dir)
#
#     args.model_dir = model_root_dir
#     args.model_path = os.path.join(args.model_dir, 'model.pt')
#
#     print(f'Experiment saved to: {args.log_dir}')
#
#     hparam_dict = {}
#
#     for k, v in vars(args).items():
#         if isinstance(v, (int, float, str, bool, torch.Tensor)):
#             hparam_dict[k] = v
#
#     print(runner_name)
#     print(args)
#
#     return args






def init_experiment(args, runner_name=None, exp_id=None, save_logs=True, save_models=True):
    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Either generate a unique experiment ID, or use one which is passed
    if exp_id is None:

        if args.exp_name is None:
            raise ValueError("Need to specify the experiment name")
        # Unique identifier for experiment
        now = '{}_({:02d}.{:02d}.{}_|_'.format(args.exp_name, datetime.now().day, datetime.now().month,
                                               datetime.now().year) + \
              datetime.now().strftime("%S.%f")[:-3] + ')'

        log_dir = os.path.join(root_dir, 'log', now)
        while os.path.exists(log_dir):
            now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
                  datetime.now().strftime("%S.%f")[:-3] + ')'

            log_dir = os.path.join(root_dir, 'log', now)

    else:

        log_dir = os.path.join(root_dir, 'log', f'{exp_id}')

    if save_logs:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger.add(os.path.join(log_dir, 'log.txt'))
        args.logger = logger
        args.log_dir = log_dir

    # Instantiate directory to save models to
    if save_models:
        model_root_dir = os.path.join(args.log_dir, 'checkpoints')
        if not os.path.exists(model_root_dir):
            os.mkdir(model_root_dir)

        args.model_dir = model_root_dir
        args.model_path = os.path.join(args.model_dir, 'model.pt')

    if save_logs:
        print(f'Experiment saved to: {args.log_dir}')

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    if save_logs:
        print(runner_name)
        print(args)

    return args






class DistributedWeightedSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, dataset, weights, num_samples, num_replicas=None, rank=None,
                 replacement=True, generator=None):
        super(DistributedWeightedSampler, self).__init__(dataset, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.weights = self.weights[self.rank::self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor = self.rank + rand_tensor * self.num_replicas
        yield from iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples


import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


class NNBatchSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
    """

    def __init__(self, data_source, model, seen_dataloader, batch_size, nn_per_image=5, using_feat=True, is_norm=False):
        self.batch_size = batch_size
        self.nn_per_image = nn_per_image
        self.using_feat = using_feat
        self.is_norm = is_norm
        self.num_samples = data_source.__len__()
        self.nn_matrix, self.dist_matrix = self._build_nn_matrix(model, seen_dataloader)

    def __iter__(self):
        for _ in range(len(self)):
            yield self.sample_batch()

    def _predict_batchwise(self, model, seen_dataloader):
        device = "cuda"
        model_is_training = model.training
        model.eval()

        ds = seen_dataloader.dataset
        A = [[] for i in range(len(ds[0]))]
        with torch.no_grad():
            # extract batches (A becomes list of samples)
            for batch in tqdm(seen_dataloader):
                for i, J in enumerate(batch):
                    # i = 0: sz_batch * images
                    # i = 1: sz_batch * labels
                    # i = 2: sz_batch * indices
                    # i = 3: sz_batch * mask_lab
                    if i == 0:
                        J = J[0]
                        # move images to device of model (approximate device)
                        # if self.using_feat:
                        #     J, _ = model(J.cuda())
                        # else:
                        J = model(J.cuda())

                        if self.is_norm:
                            J = F.normalize(J, p=2, dim=1)

                    for j in J:
                        A[i].append(j)

        model.train()
        model.train(model_is_training)  # revert to previous training state

        return [torch.stack(A[i]) for i in range(len(A))]

    def _build_nn_matrix(self, model, seen_dataloader):
        # calculate embeddings with model and get targets
        X, T, _, _ = self._predict_batchwise(model, seen_dataloader)

        # get predictions by assigning nearest 8 neighbors with cosine
        K = self.nn_per_image * 1
        nn_matrix = []
        dist_matrix = []
        xs = []

        for x in X:
            if len(xs) < 5000:
                xs.append(x)
            else:
                xs.append(x)
                xs = torch.stack(xs, dim=0)

                dist_emb = xs.pow(2).sum(1) + (-2) * X.mm(xs.t())
                dist_emb = X.pow(2).sum(1) + dist_emb.t()

                ind = dist_emb.topk(K, largest=False)[1].long().cpu()
                dist = dist_emb.topk(K, largest=False)[0]
                nn_matrix.append(ind)
                dist_matrix.append(dist.cpu())
                xs = []
                del ind

        # Last Loop
        xs = torch.stack(xs, dim=0)
        dist_emb = xs.pow(2).sum(1) + (-2) * X.mm(xs.t())
        dist_emb = X.pow(2).sum(1) + dist_emb.t()
        ind = dist_emb.topk(K, largest=False)[1]
        dist = dist_emb.topk(K, largest=False)[0]
        nn_matrix.append(ind.long().cpu())
        dist_matrix.append(dist.cpu())
        nn_matrix = torch.cat(nn_matrix, dim=0)
        dist_matrix = torch.cat(dist_matrix, dim=0)

        return nn_matrix, dist_matrix

    def sample_batch(self):
        num_image = self.batch_size // self.nn_per_image
        sampled_queries = np.random.choice(self.num_samples, num_image, replace=False)
        # 检查索引范围
        # if max(sampled_queries) >= self.nn_matrix.size(0):
        #     raise ValueError(
        #         f"Sampled query index out of bounds. Max index: {max(sampled_queries)}, NN matrix size: {self.nn_matrix.size(0)}")
        sampled_indices = self.nn_matrix[sampled_queries].view(-1).tolist()

        return sampled_indices

    def __len__(self):
        return self.num_samples // self.batch_size


import torch.nn as nn


class RC_STML(nn.Module):
    def __init__(self, sigma, delta, view, disable_mu, topk):
        super(RC_STML, self).__init__()
        self.sigma = sigma
        self.delta = delta
        self.view = view
        self.disable_mu = disable_mu
        self.topk = topk

    def k_reciprocal_neigh(self, initial_rank, i, topk):
        forward_k_neigh_index = initial_rank[i, :topk]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :topk]
        fi = np.where(backward_k_neigh_index == i)[0]
        return forward_k_neigh_index[fi]

    def forward(self, s_emb, t_emb, idx, v2=False):
        if v2:
            return self.forward_v2(t_emb, s_emb)
        if self.disable_mu:
            s_emb = F.normalize(s_emb)
        t_emb = F.normalize(t_emb)

        N = len(s_emb)
        S_dist = torch.cdist(s_emb, s_emb)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)

        with torch.no_grad():
            T_dist = torch.cdist(t_emb, t_emb)
            W_P = torch.exp(-T_dist.pow(2) / self.sigma)

            batch_size = len(s_emb) // self.view
            W_P_copy = W_P.clone()
            W_P_copy[idx.unsqueeze(1) == idx.unsqueeze(1).t()] = 1

            topk_index = torch.topk(W_P_copy, self.topk)[1]
            topk_half_index = topk_index[:, :int(np.around(self.topk / 2))]

            W_NN = torch.zeros_like(W_P).scatter_(1, topk_index, torch.ones_like(W_P))
            V = ((W_NN + W_NN.t()) / 2 == 1).float()

            W_C_tilda = torch.zeros_like(W_P)
            for i in range(N):
                indNonzero = torch.where(V[i, :] != 0)[0]
                W_C_tilda[i, indNonzero] = (V[:, indNonzero].sum(1) / len(indNonzero))[indNonzero]

            W_C_hat = W_C_tilda[topk_half_index].mean(1)
            W_C = (W_C_hat + W_C_hat.t()) / 2
            W = (W_P + W_C) / 2

            identity_matrix = torch.eye(N).cuda(non_blocking=True)
            pos_weight = (W) * (1 - identity_matrix)
            neg_weight = (1 - W) * (1 - identity_matrix)

        pull_losses = torch.relu(S_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - S_dist).pow(2) * neg_weight

        loss = (pull_losses.sum() + push_losses.sum()) / (len(s_emb) * (len(s_emb) - 1))

        return loss

    def forward_v2(self, probs, feats):
        with torch.no_grad():
            pseudo_labels = probs.argmax(1).cuda()
            one_hot = torch.zeros(probs.shape).cuda().scatter(1, pseudo_labels.unsqueeze(1), 1.0)
            W_P = torch.mm(one_hot, one_hot.t())
            feats_dist = torch.cdist(feats, feats)
            topk_index = torch.topk(feats_dist, self.topk)[1]
            W_NN = torch.zeros_like(feats_dist).scatter_(1, topk_index, W_P)

            W = ((W_NN + W_NN.t()) / 2 == 0.5).float()

            N = len(probs)
            identity_matrix = torch.eye(N).cuda(non_blocking=True)
            pos_weight = (W) * (1 - identity_matrix)
            neg_weight = (1 - W) * (1 - identity_matrix)

        pull_losses = torch.relu(feats_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - feats_dist).pow(2) * neg_weight
        # pull_losses = F.smooth_l1_loss(feats_dist, torch.zeros_like(feats_dist), reduction='none') * pos_weight
        # push_losses = F.smooth_l1_loss(self.delta - feats_dist, torch.zeros_like(feats_dist),
        #                                reduction='none') * neg_weight

        loss = (pull_losses.sum() + push_losses.sum()) / (len(probs) * (len(probs) - 1))

        return loss


class KL_STML(nn.Module):
    def __init__(self, disable_mu, temp=1):
        super(KL_STML, self).__init__()
        self.disable_mu = disable_mu
        self.temp = temp

    def kl_div(self, A, B, T=1):
        log_q = F.log_softmax(A / T, dim=-1)
        p = F.softmax(B / T, dim=-1)
        kl_d = F.kl_div(log_q, p, reduction='sum') * T ** 2 / A.size(0)
        return kl_d

    def forward(self, s_f, s_g):
        if self.disable_mu:
            s_f, s_g = F.normalize(s_f), F.normalize(s_g)

        N = len(s_f)
        S_dist = torch.cdist(s_f, s_f)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)

        S_bg_dist = torch.cdist(s_g, s_g)
        S_bg_dist = S_bg_dist / S_bg_dist.mean(1, keepdim=True)

        loss = self.kl_div(-S_dist, -S_bg_dist.detach(), T=1)

        return loss


class STML_loss(nn.Module):
    def __init__(self, sigma, delta, view, disable_mu, topk):
        super(STML_loss, self).__init__()
        self.sigma = sigma
        self.delta = delta
        self.view = view
        self.disable_mu = disable_mu
        self.topk = topk
        self.RC_criterion = RC_STML(sigma, delta, view, disable_mu, topk)
        self.KL_criterion = KL_STML(disable_mu, temp=1)

    def forward(self, s_f, s_g, t_g, idx):
        # Relaxed contrastive loss for STML
        loss_RC_f = self.RC_criterion(s_f, t_g, idx)
        loss_RC_g = self.RC_criterion(s_g, t_g, idx)
        loss_RC = (loss_RC_f + loss_RC_g) / 2

        # Self-Distillation for STML
        loss_KL = self.KL_criterion(s_f, s_g)

        loss = loss_RC + loss_KL

        total_loss = dict(RC=loss_RC, KL=loss_KL, loss=loss)

        return total_loss


class STML_loss_simgcd(nn.Module):
    def __init__(self, disable_mu=0, topk=4, sigma=1, delta=1, view=2, v2=True):
        super().__init__()
        self.sigma = sigma
        self.delta = delta
        self.view = view
        self.disable_mu = disable_mu
        self.topk = topk
        self.v2 = v2
        self.RC_criterion = RC_STML(sigma, delta, view, disable_mu, topk)
        self.KL_criterion = KL_STML(disable_mu, temp=1)

    def forward(self, s_g, t_g, idx):
        # Relaxed contrastive loss for STML
        loss_RC = self.RC_criterion(s_g, t_g, idx, v2=self.v2)

        if not self.v2:
            # Self-Distillation for STML
            loss_KL = self.KL_criterion(s_g, t_g)
        else:
            loss_KL = 0.0
        loss = loss_RC + loss_KL

        # total_loss = dict(RC=loss_RC, KL=loss_KL, loss=loss)

        return loss

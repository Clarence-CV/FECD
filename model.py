import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Any, Tuple
import numpy as np
import torch.nn as nn
from torch.autograd import Function
import torch


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

    
    
    
    



def shift_log(x, offset=1e-6):
    """
    First shift, then calculate log for numerical stability.
    """

    return torch.log(torch.clamp(x + offset, max=1.))




class WorstCaseEstimationLoss(nn.Module):

    def __init__(self, eta_prime):
        super(WorstCaseEstimationLoss, self).__init__()
        self.eta_prime = eta_prime

    def forward(self, y_l, y_l_adv, y_u, y_u_adv):
        _, prediction_l = y_l.max(dim=1)
        loss_l = self.eta_prime * F.cross_entropy(y_l_adv, prediction_l)

        _, prediction_u = y_u.max(dim=1)
        loss_u = F.nll_loss(shift_log(1. - F.softmax(y_u_adv, dim=1)), prediction_u)

        return loss_l + loss_u











class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)





class WarmStartGradientReverseLayer(nn.Module):
    """
    Gradient reversal layer that can be warm started.
    """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float64(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1


class DINOHeadExtended(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, grl_alpha=0.1):
        super(DINOHeadExtended, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.use_bn = use_bn
        self.norm_last_layer = norm_last_layer

        # Main MLP layers
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        # Main head
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        # Auxiliary head
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=grl_alpha, max_iters=1000, auto_step=False)
        self.aux_head = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x_proj = self.mlp(x)
        x_norm = F.normalize(x_proj, dim=-1, p=2)


        main_logits = self.last_layer(x_norm)


        reversed_features = self.grl_layer(x_proj)
        aux_logits = self.aux_head(reversed_features)
        return x_proj, main_logits, aux_logits
    
    

class DynamicLSR(nn.Module):
    '''
    Dynamic version of LSR
    '''
    def __init__(self, initial_e, total_steps):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.initial_e = initial_e
        self.total_steps = total_steps
        self.step = 0

    def _one_hot(self, labels, classes, value=1):
        one_hot = torch.zeros(labels.size(0), classes, device=labels.device)
        labels = labels.view(labels.size(0), -1)
        one_hot.scatter_(1, labels, value)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor, class_weights):
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        smooth_value = (smooth_factor / length) * class_weights
        one_hot += smooth_value
        return one_hot

    def update_smoothing_factor(self):
        # Linear decay of the smoothing factor
        e = self.initial_e * (1 - self.step / self.total_steps)
        self.step += 1
        return e

    def adjust_weights(self, outputs, targets):
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            correct = preds.eq(targets).float()
            class_accuracy = torch.bincount(targets, weights=correct, minlength=outputs.size(1))
            class_totals = torch.bincount(targets, minlength=outputs.size(1))
            class_accuracy = class_accuracy / class_totals.clamp(min=1)  # Prevent division by zero
            class_weights = 1.0 / (class_accuracy + 1e-5)  # Smooth to avoid division by zero
        return class_weights / class_weights.sum()

    def forward(self, x, target):
        e = self.update_smoothing_factor()
        class_weights = self.adjust_weights(x.detach(), target)
        smoothed_target = self._smooth_label(target, x.size(1), e, class_weights)
        x = self.log_softmax(x)
        loss = torch.sum(-x * smoothed_target, dim=1)
        return torch.mean(loss)
    
    
    
    

class WBRegularizationLoss(nn.Module):
    '''

    Constrain the cluster
    '''
    def __init__(self, num_classes, feat_dim, device):
        super(WBRegularizationLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

    def forward(self, features, labels):
        unique_labels = torch.unique(labels)
        global_mean = torch.mean(features, dim=0)

        W = 0
        B = 0
        total_points = features.shape[0]

        for label in unique_labels:
            class_mask = (labels == label)
            class_features = features[class_mask]
            class_mean = torch.mean(class_features, dim=0)
            # Intra-class variance (W)
            W += torch.sum((class_features - class_mean) ** 2)
            # Inter-class variance (B)
            B += class_mask.sum() * torch.sum((class_mean - global_mean) ** 2)

        wb_ratio = W / (B + 1e-8)  #\theta
        return wb_ratio


# class MaxMinRegularizationLoss(nn.Module):
#     def __init__(self, num_classes, feat_dim, device):
#         super(MaxMinRegularizationLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.device = device

#     def forward(self, features, labels):
#         unique_labels = torch.unique(labels)
#         max_min_loss = 0.0

#         for label in unique_labels:
#             class_mask = (labels == label)
#             class_features = features[class_mask]
#             class_center = torch.mean(class_features, dim=0, keepdim=True)

#             # Calculate the distances from each point to the class center
#             distances = torch.norm(class_features - class_center, p=2, dim=1)
#             max_distance = torch.max(distances)
#             min_distance = torch.min(distances)

#             # Max-Min distance difference within the class
#             max_min_loss += (max_distance - min_distance)

#         return max_min_loss / len(unique_labels)
class MaxMinRegularizationLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(MaxMinRegularizationLoss, self).__init__() 
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

    def forward(self, features, labels):
        unique_labels = torch.unique(labels)
        max_min_loss = 0.0

        for label in unique_labels:
            class_mask = (labels == label)
            class_features = features[class_mask]
            class_center = torch.mean(class_features, dim=0, keepdim=True)

            # Calculate the squared distances from each point to the class center
            distances = torch.sum((class_features - class_center) ** 2, dim=1)
            max_distance = torch.max(distances)
            min_distance = torch.min(distances)

            # Max-Min squared distance difference within the class
            max_min_loss += (max_distance - min_distance)

        return max_min_loss / len(unique_labels)


class TotalClusteringLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device, wb_weight=1.0, max_min_weight=0.1):
        super(TotalClusteringLoss, self).__init__()
        self.wb_loss = WBRegularizationLoss(num_classes, feat_dim, device)
        self.max_min_loss = MaxMinRegularizationLoss(num_classes, feat_dim, device)
        self.wb_weight = wb_weight
        self.max_min_weight = max_min_weight

    def forward(self, features, labels):
        wb_ratio_loss = self.wb_loss(features, labels)
        max_min_reg_loss = self.max_min_loss(features, labels)
        total_loss = (self.wb_weight * wb_ratio_loss +
                      self.max_min_weight * max_min_reg_loss)
        return total_loss



    

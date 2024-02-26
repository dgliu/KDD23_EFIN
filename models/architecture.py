import numpy as np

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


class EFIN(nn.Module):
    """
    EFIN class -- a explicit feature interaction network with two heads.
    """
    def __init__(self, input_dim, hc_dim, hu_dim, is_self, act_type='elu'):
        super(EFIN, self).__init__()
        self.nums_feature = input_dim
        self.is_self = is_self

        # interaction attention
        self.att_embed_1 = nn.Linear(hu_dim, hu_dim, bias=False)
        self.att_embed_2 = nn.Linear(hu_dim, hu_dim)
        self.att_embed_3 = nn.Linear(hu_dim, 1, bias=False)

        # self-attention
        self.softmax = nn.Softmax(dim=-1)
        self.Q_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)
        self.K_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)
        self.V_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)

        # representation parts for X
        self.x_rep = nn.Embedding(input_dim, hu_dim)

        # representation parts for T
        self.t_rep = nn.Linear(1, hu_dim)

        '''control net'''
        self.c_fc1 = nn.Linear(input_dim * hu_dim, hc_dim)
        self.c_fc2 = nn.Linear(hc_dim, hc_dim)
        self.c_fc3 = nn.Linear(hc_dim, hc_dim // 2)
        self.c_fc4 = nn.Linear(hc_dim // 2, hc_dim // 4)
        out_dim = hc_dim // 4
        if self.is_self:
            self.c_fc5 = nn.Linear(hc_dim / 4, hc_dim // 8)
            out_dim = hc_dim // 8

        self.c_logit = nn.Linear(out_dim, 1)
        self.c_tau = nn.Linear(out_dim, 1)

        '''uplift net'''
        self.u_fc1 = nn.Linear(hu_dim, hu_dim)
        self.u_fc2 = nn.Linear(hu_dim, hu_dim // 2)
        self.u_fc3 = nn.Linear(hu_dim // 2, hu_dim // 4)
        out_dim = hu_dim // 4
        if self.is_self:
            self.u_fc4 = nn.Linear(hu_dim // 4, hu_dim // 8)
            out_dim = hu_dim // 8
        self.t_logit = nn.Linear(out_dim, 1)
        self.u_tau = nn.Linear(out_dim, 1)

        # activation function
        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        else:
            raise RuntimeError('unknown act_type {0}'.format(act_type))

    def self_attn(self, q, k, v):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)
        attn_weights = Q.matmul(torch.transpose(K, 1, 2)) / (K.shape[-1] ** 0.5)
        attn_weights = self.softmax(torch.sigmoid(attn_weights))

        outputs = attn_weights.matmul(V)

        return outputs, attn_weights

    def interaction_attn(self, t, x):
        attention = []
        for i in range(self.nums_feature):
            temp = self.att_embed_3(torch.relu(
                torch.sigmoid(self.att_embed_1(t)) + torch.sigmoid(self.att_embed_2(x[:, i, :]))))
            attention.append(temp)
        attention = torch.squeeze(torch.stack(attention, 1), 2)
        # print('interaction attention', attention)
        attention = torch.softmax(attention, 1)
        # print('mean interaction attention', torch.mean(attention, 0))

        outputs = torch.squeeze(torch.matmul(torch.unsqueeze(attention, 1), x), 1)
        return outputs, attention

    def forward(self, feature_list, is_treat):
        t_true = torch.unsqueeze(is_treat, 1)

        x_rep = feature_list.unsqueeze(2) * self.x_rep.weight.unsqueeze(0)

        # control net
        dims = x_rep.size()
        _x_rep = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True)
        xx, xx_weight = self.self_attn(_x_rep, _x_rep, _x_rep)

        _x_rep = torch.reshape(xx, (dims[0], dims[1] * dims[2]))

        c_last = self.act(self.c_fc4(self.act(self.c_fc3(self.act(self.c_fc2(self.act(self.c_fc1(_x_rep))))))))
        if self.is_self:
            c_last = self.act(self.c_fc5(c_last))
        c_logit = self.c_logit(c_last)
        c_tau = self.c_tau(c_last)
        c_prob = torch.sigmoid(c_logit)

        # uplift net
        t_rep = self.t_rep(torch.ones_like(t_true))

        xt, xt_weight = self.interaction_attn(t_rep, x_rep)

        u_last = self.act(self.u_fc3(self.act(self.u_fc2(self.act(self.u_fc1(xt))))))
        if self.is_self:
            u_last = self.act(self.u_fc4(u_last))
        t_logit = self.t_logit(u_last)
        u_tau = self.u_tau(u_last)
        t_prob = torch.sigmoid(t_logit)

        return c_logit, c_prob, c_tau, t_logit, t_prob, u_tau

    def calculate_loss(self, feature_list, is_treat, label_list):
        # Model outputs
        c_logit, c_prob, c_tau, t_logit, t_prob, u_tau = self.forward(feature_list, is_treat)

        # regression
        c_logit_fix = c_logit.detach()
        uc = c_logit
        ut = (c_logit_fix + u_tau)

        y_true = torch.unsqueeze(label_list, 1)
        t_true = torch.unsqueeze(is_treat, 1)

        # response loss
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        temp = torch.square((1 - t_true) * uc + t_true * ut - y_true)
        loss1 = torch.mean(temp)
        loss2 = criterion(t_logit, 1 - t_true)
        loss = loss1 + loss2

        return loss
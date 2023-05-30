#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import numpy as np
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")



class CausalVAE(nn.Module):
    def __init__(self, nn='mask', name='vae', z_dim=16, z1_dim=4, z2_dim=4, inference = False, alpha=0.3, beta=1):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        # 猜测：z_dim是整体的latent维度，z1_dim是concept的个数，z2_dim是每个concept的隐空间的维度
        self.channel = 4
        self.scale = np.array([[0,44],[100,40],[6.5, 3.5],[10,5]]) # 初始生成pendulum数据的每个concept的scale
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        # 这里CausalVAE内部的nn把外部的nn给屏蔽掉了
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.channel)
        self.dec = nn.Decoder_DAG(self.z_dim,self.z1_dim, self.z2_dim)
        self.dag = nn.DagLayer(self.z1_dim, self.z1_dim, i = inference) # in_features, out_features都是z1_dim
        #self.cause = nn.CausalLayer(self.z_dim, self.z1_dim, self.z2_dim)
        self.attn = nn.Attention(self.z2_dim)
        self.mask_z = nn.MaskLayer(self.z_dim, z1_dim=self.z2_dim)
        self.mask_u = nn.MaskLayer(self.z1_dim,z1_dim=1)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x, label, mask = None, sample = False, adj = None, alpha=0.3, beta=1, lambdav=0.001):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        assert label.size()[1] == self.z1_dim

        q_m, q_v = self.enc.encode(x.to(device))
        # 只传入x，不传入y 4*96*96-900-300-300-2*16
        # q_m比较接近向量0，q_v比较接近向量1，因为q_v经过了一个softplus
        # q_m, q_v [64, 16]
        # breakpoint()
        # z1 z2是什么？
        q_m, q_v = q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)
        # 为什么要直接把q_v变成全1呢？
        # q_m, q_v [64, 4, 4]

        decode_m, decode_v = self.dag.calculate_dag(q_m.to(device), torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device))
        # decode_v是一个全1矩阵，在calculate_dag的过程中没有任何变化
        # decode_m, decode_v [64, 4, 4]
        decode_m, decode_v = decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),decode_v
        if sample == False:
          if mask != None and mask < 2:
              z_mask = torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)*adj
              decode_m[:, mask, :] = z_mask[:, mask, :]
              decode_v[:, mask, :] = z_mask[:, mask, :]
          m_zm, m_zv = self.dag.mask_z(decode_m.to(device)).reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),decode_v.reshape([q_m.size()[0], self.z1_dim,self.z2_dim])
          # m_zm, m_zv [64, 4, 4]
          # shape似乎没有在reshape中有任何变化。如果z1 z2不一样呢？
          m_u = self.dag.mask_u(label.to(device))
          # m_u [64, 4, 1]
          # 第一次mask结束后除了所有的变量不是0就是1，怎么反向传播呢？
          
          f_z = self.mask_z.mix(m_zm).reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device)
          # 这里本来每个concept只用4个dim，然后每个concept分别过了一个网络，再回到4dim，增强表示能力？
          # 这一步结束之后，不再是全0值了，但是这一步并没有用到m_zv
          # f_z [64, 4, 4]
          e_tilde = self.attn.attention(decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device),q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device))[0]
          # e_tilde [64, 4, 4]
          if mask != None and mask < 2:
              z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
              e_tilde[:, mask, :] = z_mask[:, mask, :]
              
          f_z1 = f_z+e_tilde
          if mask!= None and mask == 2 :
              z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
              f_z1[:, mask, :] = z_mask[:, mask, :]
              m_zv[:, mask, :] = z_mask[:, mask, :]
          if mask!= None and mask == 3 : # 根据mask = 0 1 2 3 4，决定进到哪一个if中去，这里的mask应该指的是在哪个阶段mask
              z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
              f_z1[:, mask, :] = z_mask[:, mask, :]
              m_zv[:, mask, :] = z_mask[:, mask, :]
          g_u = self.mask_u.mix(m_u).to(device) # 和图像一样的进一个mix，到mix函数的第一个if里面去，同样过几个独立的神经网络
          z_given_dag = ut.conditional_sample_gaussian(f_z1, m_zv*lambdav) # lambdav为什么这么小
        
        decoded_bernoulli_logits,x1,x2,x3,x4 = self.dec.decode_sep(z_given_dag.reshape([z_given_dag.size()[0], self.z_dim]), label.to(device))
        # label没有被使用到
        # [64, 4, 4] - [64, 16] - 4 * [64, 4] - 4 * [64, 300] - 300 - 1024 - 4 * 96 * 96
        # 这份代码一定经过多次迭代，且维护的不算好

        # rec = ut.log_bernoulli_with_logits(x, decoded_bernoulli_logits.reshape(x.size()))
        rec = ut.log_bernoulli_with_logits(x.reshape(decoded_bernoulli_logits.size()), decoded_bernoulli_logits)
        rec = -torch.mean(rec)

        p_m, p_v = torch.zeros(q_m.size()), torch.ones(q_m.size())
        cp_m, cp_v = ut.condition_prior(self.scale, label, self.z2_dim)
        cp_v = torch.ones([q_m.size()[0],self.z1_dim,self.z2_dim]).to(device)
        cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))
        kl = torch.zeros(1).to(device)
        kl = alpha*ut.kl_normal(q_m.view(-1,self.z_dim).to(device), q_v.view(-1,self.z_dim).to(device), p_m.view(-1,self.z_dim).to(device), p_v.view(-1,self.z_dim).to(device))
        # 计算q和p的kl散度
        # kl: [64,]

        for i in range(self.z1_dim):
            kl = kl + beta*ut.kl_normal(decode_m[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
        # 计算decode和c的kl散度（c指的是conditional prior）
        kl = torch.mean(kl)
        mask_kl = torch.zeros(1).to(device)
        mask_kl2 = torch.zeros(1).to(device)

        for i in range(4):
            mask_kl = mask_kl + 1*ut.kl_normal(f_z1[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
        # 计算f_z1和cp的kl散度
        
        u_loss = torch.nn.MSELoss()
        mask_l = torch.mean(mask_kl) + u_loss(g_u, label.float().to(device))
        # 计算g_u和label的MSELoss
        nelbo = rec + kl + mask_l

        return nelbo, kl, rec, decoded_bernoulli_logits.reshape(x.size()), z_given_dag

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))

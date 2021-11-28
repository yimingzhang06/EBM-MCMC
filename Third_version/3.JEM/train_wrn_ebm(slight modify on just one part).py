#this code is refered from the author's official code, and the URL is : https://github.com/wgrathwohl/JEM
#In the <train_wrn_ebm.py>, I insert some functions and codes to try new sampling methods.
#1. update the get_sample_q function slightly to try different sampling methods.
#2. insert sgld_sampling function to try sgld sampling.
#3. insert mala_sampling function to try mala.
#4. insert psgld_sampling function to try psgld.
#5. insert run_rhat to calculate the r-hat value to evaluate the MCMC effect.

#All other .py files have not changed because there are no relations between other files and sampling.




# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import utils
import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
#import ipdb
import numpy as np
import wideresnet
import json
import matplotlib.pyplot as plt
# Sampling
from tqdm import tqdm
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
im_sz = 32
n_ch = 3

energy_difference_total = []
x_gradient_total = []

e_pos_total = []
e_pos_std_total = []

e_neg_total = []
e_neg_std_total = []

L_list = []

l_p_x_list = []
acc_list = []
l_p_y_given_x_list = []

acceptance_ratio = []

sample_method = []



class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


class CCF(F):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=n_classes)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


def cycle(loader):
    while True:
        for data in loader:
            yield data


def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    total_norm = total_norm ** (1. / 2)
    return total_norm.item()


def grad_vals(m):
    ps = []
    for p in m.parameters():
        if p.grad is not None:
            ps.append(p.grad.data.view(-1))
    ps = t.cat(ps)
    return ps.mean().item(), ps.std(), ps.abs().mean(), ps.abs().std(), ps.abs().min(), ps.abs().max()


def init_random(args, bs):
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def get_model_and_buffer(args, device, sample_q):
    model_cls = F if args.uncond else CCF
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]

    f = f.to(device)
    return f, replay_buffer


def get_data(args):
    if args.dataset == "svhn":
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    else:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + args.sigma * t.randn_like(x)]
    )
    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        else:
            return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")

    # get all training inds
    full_train = dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(1234)
    # shuffle
    np.random.shuffle(all_inds)
    # seperate out validation set
    if args.n_valid is not None:
        valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    train_inds = np.array(train_inds)
    train_labeled_inds = []
    other_inds = []
    train_labels = np.array([full_train[ind][1] for ind in train_inds])
    if args.labels_per_class > 0:
        for i in range(args.n_classes):
            print(i)
            train_labeled_inds.extend(train_inds[train_labels == i][:args.labels_per_class])
            other_inds.extend(train_inds[train_labels == i][args.labels_per_class:])
    else:
        train_labeled_inds = train_inds

    dset_train = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_inds)
    dset_train_labeled = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_labeled_inds)
    dset_valid = DataSubset(
        dataset_fn(True, transform_test),
        inds=valid_inds)
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dset_test = dataset_fn(False, transform_test)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    return dload_train, dload_train_labeled, dload_valid,dload_test


def get_sample_q(args, device):
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(args, bs), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
        inds = t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(args, bs)
        choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None, None, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(device), inds

    def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps, test = False, iter = 0, every = 100, rhat = False, device = device):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        bs = args.batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        if (args.sample == 'sgld'):
            if rhat == True:
                x_k, all_sample = sgld_sampling(args, f, x_k, y, device, test = test, iter = iter, every = every, rhat = rhat)
            else:
                x_k = sgld_sampling(args, f, x_k, y, device, test = test, iter = iter, every = every, rhat = rhat)
        elif (args.sample == 'mala'):
            if rhat == True:
                x_k, all_sample = mala_sampling(args, f, x_k, y, device, test = test, iter = iter, every = every, rhat = rhat)
            else:
                x_k = mala_sampling(args, f, x_k, y, device, test = test, iter = iter, every = every, rhat = rhat)
          
        elif (args.sample == 'psgld'):
            if rhat == True:
                x_k, all_sample = psgld_sampling(args, f, x_k, y, device, test = test, iter = iter, every = every, rhat = rhat)
            else:
                x_k = psgld_sampling(args, f, x_k, y, device, test = test, iter = iter, every = every, rhat = rhat)
        else:
          print("error sampling method")


        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        if rhat == False:
            return final_samples
        else:
            return final_samples, all_sample

    return sample_q

# args.tem
# args.mala_variance
# args.sample
 
def sgld_sampling(args, energy, x_k, y, device, test = False, iter = 0, every = 100, rhat = False):

    ac_list = []
    q_list = []
    energy_diff_list = []
    move_list = []
    grad_list = []
    energy_list = []

    all_sample = []

    x_grad = 0
    ra = 0

    for k in range(args.n_steps):

        x_k_original = x_k

        en = energy(x_k, y=y)/args.tem

        energy_grad = t.autograd.grad(
            en.sum(), [x_k], retain_graph=False)[0]

        x_k.data += args.sgld_lr * energy_grad + args.sgld_std * t.randn_like(x_k)

        x_grad = x_grad + energy_grad

        if rhat == True:
            all_sample.append(en)

        if test == True:
        
            en_2 = energy(x_k, y=y)/args.tem

            energy_grad_2 = t.autograd.grad(
                en_2.sum(), [x_k], retain_graph=False)[0]

            x2_to_x_original = 1 / (-4 * args.sgld_lr * args.mala_variance_tem) * torch.sum(((x_k.data - x_k_original.data + args.sgld_lr * energy_grad) ** 2), dim=[1,2,3])
            x_original_to_x2 = 1 / (-4 * args.sgld_lr * args.mala_variance_tem) * torch.sum(((x_k_original.data - x_k.data + args.sgld_lr * energy_grad_2) ** 2), dim=[1,2,3])                                                
        
            log_alpha = en - en_2 + x_original_to_x2 - x2_to_x_original

            q_difference = torch.mean((x_original_to_x2 - x2_to_x_original).float()).float()
            q_list.append(q_difference.item())


            energy_list.append(torch.mean(en.float()).item())

            energy_difference = torch.mean((en - en_2).float()).float()
            energy_diff_list.append(energy_difference.item())

            alpha = torch.exp(torch.clamp_max(log_alpha, 0))
            mask = torch.rand(x_k.data.shape[0], device=device)

            ratio = mask < alpha
            me_ = torch.mean(ratio.float()).float()
            ac_list.append(me_.item())
            ra += me_.item()

            square = ((x_k_original.data-x_k.data) ** 2).sum(dim=tuple(range(1, (x_k_original.data-x_k.data).ndim)))
            move_dis = (square ** (0.5)).mean()
            move_list.append(move_dis.item())

            grad_list.append(torch.mean(energy_grad.float()).item())

    if test == True:

            sample_method.append(ra/args.n_steps)
            x_gradient_total.append(x_grad.mean().item())

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,energy_list)
            
            plt.xlabel('Steps in chain')
            plt.ylabel('Energy')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"ld_no_Energy")
            plt.close(0)


            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,ac_list)
            
            plt.xlabel('Steps in chain')
            plt.ylabel('AC-ratio')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"ld_no_Accept")
            plt.close(0)

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,move_list)
            
            plt.xlabel('Steps in chain')
            plt.ylabel('Transition Distance')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"ld_no_Distance")
            plt.close(0)

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,energy_diff_list)
            
            plt.xlabel('Steps in chain')
            plt.ylabel('Energy Difference')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"ld_no_EnergyDifference")
            plt.close(0)

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,q_list)
            
            plt.xlabel('Steps in chain')
            plt.ylabel('q(X|X`) Difference')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"ld_no_QDifference")
            plt.close(0)

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,grad_list)
            
            plt.xlabel('Steps in chain')
            plt.ylabel('Gradient')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"ld_no_Gradient_energy")
            plt.close(0)
    
    
    if rhat == True:
        return x_k, all_sample
    else:
        return x_k


def mala_sampling(args, energy, x_k, y, device, test = False, iter = 0, every = 100, rhat = False):
    ac_list = []
    q_list = []
    energy_diff_list = []
    move_list = []
    grad_list = []

    all_sample = []
    energy_list = []

    x_grad = 0
    ra = 0
    for k in range(args.n_steps):
        #with torch.no_grad():
        
        x_k_original = x_k

        en = energy(x_k, y=y)/args.tem

        energy_grad = t.autograd.grad(
                en.sum(), [x_k], retain_graph=False)[0]

        x_grad = x_grad + energy_grad


        x_k.data += args.sgld_lr * energy_grad + args.sgld_std * t.randn_like(x_k)
        
        en_2 = energy(x_k, y=y)/args.tem

        energy_grad_2 = t.autograd.grad(
                en_2.sum(), [x_k], retain_graph=False)[0]

        x2_to_x_original = 1 / (-4 * args.sgld_lr * args.mala_variance_tem) * torch.sum(((x_k.data - x_k_original.data + args.sgld_lr * energy_grad) ** 2), dim=[1,2,3])
        x_original_to_x2 = 1 / (-4 * args.sgld_lr * args.mala_variance_tem) * torch.sum(((x_k_original.data - x_k.data + args.sgld_lr * energy_grad_2) ** 2), dim=[1,2,3])                                                
        
        log_alpha = en - en_2 + x_original_to_x2 - x2_to_x_original


        alpha = torch.exp(torch.clamp_max(log_alpha, 0))
        mask = torch.rand(x_k.data.shape[0], device=device)

        ratio = mask < alpha


        while len(ratio.shape) < len(x_k.data.shape):
            ratio.unsqueeze_(dim=-1)

        result = torch.where(ratio, x_k.data, x_k_original.data)
        x_k.data = result

        if rhat == True:
            all_sample.append(en)

        if test == True:
            

            energy_difference = torch.mean((en - en_2).float()).float()
            energy_diff_list.append(energy_difference.item())

            energy_list.append((torch.mean(en.float()).float()).item())

            q_difference = torch.mean((x_original_to_x2 - x2_to_x_original).float()).float()
            q_list.append(q_difference.item())

            me_ = torch.mean(ratio.float()).float()
            ac_list.append(me_.item())
            ra += me_.item()

            square = ((x_k_original.data-x_k.data) ** 2).sum(dim=tuple(range(1, (x_k_original.data-x_k.data).ndim)))
            move_dis = (square ** (0.5)).mean()
            move_list.append(move_dis.item())

            grad_list.append(torch.mean(energy_grad.float()).item())
    
    if test == True:
            sample_method.append(ra/args.n_steps)
            x_gradient_total.append(x_grad.mean().item())
            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,energy_list)        
            plt.xlabel('Steps in chain')
            plt.ylabel('Energy')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"mala_Energy")
            plt.close(0)

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,ac_list)     
            plt.xlabel('Steps in chain')
            plt.ylabel('AC-ratio')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"mala_Accept")
            plt.close(0)

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,move_list)   
            plt.xlabel('Steps in chain')
            plt.ylabel('Transition Distance')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"mala_Distance")
            plt.close(0)

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,energy_diff_list)
            plt.xlabel('Steps in chain')
            plt.ylabel('Energy Difference')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"mala_EnergyDifference")
            plt.close(0)

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,q_list)
            plt.xlabel('Steps in chain')
            plt.ylabel('q(X|X`) Difference')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"mala_QDifference")
            plt.close(0)

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,grad_list)
            plt.xlabel('Steps in chain')
            plt.ylabel('Gradient')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"mala_Gradient_energy")
            plt.close(0)

    if rhat == True:
        return x_k, all_sample
    else:
        return x_k

# args.Lamada = 1
def psgld_sampling(args, energy, x_k, y, device, test = False, iter = 0, every = 100, rhat = False):

    avg_list = []
    energy_list = []
    move_list = []
    grad_list = []

    square_avg = torch.zeros_like(x_k.data)

    all_sample = []

    x_grad = 0
    aver = 0

    for k in range(args.n_steps):
        
        x_k_original = x_k

        beta = 0.9
        
        en = energy(x_k, y=y)/args.tem

        energy_grad = t.autograd.grad(
            en.sum(), [x_k], retain_graph=False)[0]

        x_grad = x_grad + energy_grad

        square_avg = square_avg * beta + energy_grad*energy_grad * (1-beta)

        avg = 1*(square_avg.sqrt().add_(args.Lambda))

        x_k.data += args.sgld_lr * energy_grad / avg + args.sgld_std * t.randn_like(x_k)

        if rhat == True:
          all_sample.append(en)

        if test == True:
            
            square = ((x_k_original.data-x_k.data) ** 2).sum(dim=tuple(range(1, (x_k_original.data-x_k.data).ndim)))
            move_dis = (square ** (0.5)).mean()
            move_list.append(move_dis)

            grad_list.append(torch.mean(energy_grad.float()).item())

            energy_list.append(torch.mean(en.float()).item())

            avg_me = torch.mean(avg.float()).float()
            avg_list.append(avg_me.item())
            aver += avg_me.item()

            
        
    if test == True:
            sample_method.append(aver/args.n_steps)
            x_gradient_total.append(x_grad.mean().item())

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,avg_list)
            plt.xlabel('Steps in chain')
            plt.ylabel('1/G')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"psgld_Gmatrix")
            plt.close(0)

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,energy_list)
            plt.xlabel('Steps in chain')
            plt.ylabel('Energy')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"psgld_Energy")
            plt.close(0)

            plt.figure(figsize = (10,10))
            x = range(args.n_steps)
            plt.plot(x,grad_list)
            plt.xlabel('Steps in chain')
            plt.ylabel('Gradient')
            plt.grid()
            plt.savefig(args.save_dir + str(iter)+"psgld_Gradient_energy")
            plt.close(0)

    if rhat == True:
        return x_k, all_sample

    return x_k
        


def within_singal_chain_average(x, n_steps, burn_in = 0):
    #calculate the average value of a singal chain.
    sum_in_singal_chain = 0
    for i in range(n_steps - burn_in):
        sum_in_singal_chain += x[i+burn_in]
    ave_in_singal_chain = sum_in_singal_chain/(n_steps-burn_in)
    return ave_in_singal_chain

def within_singal_chain_square(x, ave_in_singal_chain, n_steps, burn_in = 0):
    #calculate the average square of a singal chain 
    sum_square = 0
    for i in range(n_steps - burn_in):
        sum_square += (x[i+burn_in] - ave_in_singal_chain) ** 2
    ave_square = sum_square/(n_steps-burn_in-1)
    return ave_square

def W_whitin_chain(all_chains_sampling, n_steps, burn_in, num_chains):
    average_each_singal_chain = []
    for i in range(num_chains):
        ave_in_singal_chain = within_singal_chain_average(all_chains_sampling[i],n_steps,burn_in)
        average_each_singal_chain.append(ave_in_singal_chain)
  
    square_each_singal_chain = []
    for i in range(num_chains):
        ave_square = within_singal_chain_square(all_chains_sampling[i], average_each_singal_chain[i],n_steps, burn_in)
        square_each_singal_chain.append(ave_square)

    square_sum_all_chain = 0
    for i in range(num_chains):
        square_sum_all_chain += square_each_singal_chain[i]
  
    square_average_all_chain = square_sum_all_chain / num_chains 

    return square_average_all_chain

def B_between_chain(all_chains_sampling, n_steps, burn_in, num_chains):

    average_each_singal_chain = []
    sum_all_chain = 0
    for i in range(num_chains):
        ave_in_singal_chain = within_singal_chain_average(all_chains_sampling[i],n_steps,burn_in)
        average_each_singal_chain.append(ave_in_singal_chain)
        sum_all_chain += average_each_singal_chain[i]
  
    average_all_chain = sum_all_chain / num_chains

    sum_square_between_chain = 0
    for i in range(num_chains):
        sum_square_between_chain += (average_each_singal_chain[i] - average_all_chain) ** 2
  
    B = sum_square_between_chain * (n_steps-burn_in) / (num_chains - 1)
  
    return B

def run_rhat(args, energy, x_k, y, test = False, iter = 0, every = 100):
    num_chains = 2
    all_chains_initial = []

    for i in range(num_chains):
        neg_x = x_k.data
        all_chains_initial.append(neg_x)
  
    n_steps = 50
    stepsize = 1e-2
    all_chains_sampling = []
    burn_in = 0

    for i in range(num_chains):
        neg_x_sampling = sample_MALA(all_chains_initial[i], Energy, stepsize, n_steps, intermediate_samples=True)
    out_i = []
    for j in range(n_steps):
        out = Energy(neg_x_sampling[j].cpu())
        out_i.append(out)
    all_chains_sampling.append(out_i)


    W = W_whitin_chain(all_chains_sampling, n_steps, burn_in, num_chains)
    B = B_between_chain(all_chains_sampling, n_steps, burn_in, num_chains)

    VAR = (((num_chains-1) / num_chains) * W + (1 / num_chains) * B)

    R = torch.sqrt(VAR/W)

    ave = sum(R,0)
    ave /= 1000

    return ave

    #num_chains = 2
    #all_chains_initial = []

    #all_chains_sampling = []
    #en_list = []
    #en_list.append(energy4rhat)
    #all_chains_sampling.append(en_list)
        



def eval_classification(f, dload, device):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        logits = f.classify(x_p_d)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


def checkpoint(f, buffer, tag, args, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)


def main(args):
    utils.makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    # datasets
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    sample_q = get_sample_q(args, device)
    f, replay_buffer = get_model_and_buffer(args, device, sample_q)

    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    # optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    cur_iter = 0

    Fid_list = []
    Is_mean_list = []
    Is_std_list = []
    correct_list = []
    loss_list = []
    

    every = args.print_every

    for epoch in range(args.n_epochs):
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))
        for i, (x_p_d, _) in tqdm(enumerate(dload_train)):
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x_p_d = x_p_d.to(device)
            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)

            L = 0.
            if args.p_x_weight > 0:  # maximize log p(x)
                if args.class_cond_p_x_sample:
                    assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                    y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                    x_q = sample_q(f, replay_buffer, y=y_q)
                else:
                    x_q = sample_q(f, replay_buffer)  # sample from log-sumexp

                fp_all = f(x_p_d)
                fq_all = f(x_q)

                fp = fp_all.mean()
                fq = fq_all.mean()
                
                l_p_x = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq, fp - fq))
                    l_p_x_list.append(l_p_x.item())
                    e_pos_total.append(fp.item())
                    e_neg_total.append(fq.item())
                    e_pos_std_total.append(fp_all.std().item())
                    e_neg_std_total.append(fq_all.std().item())

                L += args.p_x_weight * l_p_x

            if args.p_y_given_x_weight > 0:  # maximize log p(y | x)
                logits = f.classify(x_lab)
                l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,cur_iter,l_p_y_given_x.item(),acc.item()))
                    acc_list.append(acc.item())
                    l_p_y_given_x_list.append(l_p_y_given_x.mean().item())
                    e_pos_total


                L += args.p_y_given_x_weight * l_p_y_given_x
                

            if args.p_x_y_weight > 0:  # maximize log p(x, y)
                assert not args.uncond, "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
                x_q_lab = sample_q(f, replay_buffer, y=y_lab)
                fp, fq = f(x_lab, y_lab).mean(), f(x_q_lab, y_lab).mean()
                l_p_x_y = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print('P(x, y) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                      fp - fq))

                L += args.p_x_y_weight * l_p_x_y

            # break if the loss diverged...easier for poppa to run experiments this way
            # if L.abs().item() > 1e8:
            #     print("BAD BOIIIIIIIIII")
            #     1/0

            optim.zero_grad()
            L.backward()
            optim.step()
            cur_iter += 1

            if cur_iter % every == 0:
                # all_en = []
                # en_list = []

                L_list.append(L.item())
                num_chains = 4
                if args.plot_uncond:
                    if args.class_cond_p_x_sample:
                        assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                        y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                        x_q, en_rhat1 = sample_q(f, replay_buffer, y=y_q, test = True, iter = cur_iter, every = every, rhat = True)
                        # _, en_rhat2 = sample_q(f, replay_buffer, y=y_q, test = False, iter = cur_iter, every = every, rhat = True)
                        # _, en_rhat3 = sample_q(f, replay_buffer, y=y_q, test = False, iter = cur_iter, every = every, rhat = True)
                        # _, en_rhat4 = sample_q(f, replay_buffer, y=y_q, test = False, iter = cur_iter, every = every, rhat = True)
                        
                    else:
                        x_q, en_rhat1 = sample_q(f, replay_buffer, test = True, iter = cur_iter, every = every, rhat = True)
                        # _, en_rhat2 = sample_q(f, replay_buffer, test = False, iter = cur_iter, every = every, rhat = True)
                        # _, en_rhat3 = sample_q(f, replay_buffer, test = False, iter = cur_iter, every = every, rhat = True)
                        # _, en_rhat4 = sample_q(f, replay_buffer, test = False, iter = cur_iter, every = every, rhat = True)
  
                    plot('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                    plot('{}/x_p_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_lab)
                if args.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                    x_q_y, en_rhat1 = sample_q(f, replay_buffer, y=y, test = True, iter = cur_iter, every = every, rhat = True)
                    # _, en_rhat2 = sample_q(f, replay_buffer, y=y, test = True, iter = cur_iter, every = every, rhat = True)
                    # _, en_rhat3 = sample_q(f, replay_buffer, y=y, test = True, iter = cur_iter, every = every, rhat = True)
                    # _, en_rhat4 = sample_q(f, replay_buffer, y=y, test = True, iter = cur_iter, every = every, rhat = True)

                    plot('{}/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)
                    plot('{}/x_p_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_lab)
                    x_q = x_q_y

                ####plot####

                #plot accuracy
                plt.figure(figsize = (10,10))
                x = range(cur_iter//every)
                plt.plot(x,acc_list)
                plt.xlabel('Iteration / '+str(every))
                plt.ylabel('Accuracy')
                plt.grid()
                plt.savefig(args.save_dir + str(cur_iter)+"accuracy")
                plt.close(0)

                #plot Loss
                plt.figure(figsize = (10,10))
                x = range(cur_iter//every)
                plt.plot(x,L_list)
                plt.xlabel('Iteration / '+str(every))
                plt.ylabel('Loss')
                plt.grid()
                plt.savefig(args.save_dir + str(cur_iter)+"Loss")
                plt.close(0)

                #plot difference p & q
                plt.figure(figsize = (10,10))
                x = range(cur_iter//every)
                plt.plot(x,l_p_x_list)
                plt.xlabel('Iteration / '+str(every))
                plt.ylabel('Energy difference')
                plt.grid()
                plt.savefig(args.save_dir + str(cur_iter)+"Energy Difference")
                plt.close(0)

                #plot x_gradient
                plt.figure(figsize = (10,10))
                x = range(cur_iter//every)
                plt.plot(x,x_gradient_total)
                plt.xlabel('Iteration / '+str(every))
                plt.ylabel('Sampling Gradient')
                plt.grid()
                plt.savefig(args.save_dir + str(cur_iter)+"Energy Gradience")
                plt.close(0)

                #plot e_pos_mean
                plt.figure(figsize = (10,10))
                x = range(cur_iter//every)
                plt.plot(x,e_pos_total)
                plt.xlabel('Iteration / '+str(every))
                plt.ylabel('postive energy')
                plt.grid()
                plt.savefig(args.save_dir + str(cur_iter)+"postive energy")
                plt.close(0)

                # plot e_pos_std
                plt.figure(figsize = (10,10))
                x = range(cur_iter//every)
                plt.plot(x,e_pos_std_total)
                plt.xlabel('Iteration / '+str(every))
                plt.ylabel('postive energy std')
                plt.grid()
                plt.savefig(args.save_dir + str(cur_iter)+"postive energy std")
                plt.close(0)

                #plot e_neg_mean
                plt.figure(figsize = (10,10))
                x = range(cur_iter//every)
                plt.plot(x,e_neg_total)
                plt.xlabel('Iteration / '+str(every))
                plt.ylabel('negative energy')
                plt.grid()
                plt.savefig(args.save_dir + str(cur_iter)+"negative energy")
                plt.close(0)

                #plot e_neg_std
                plt.figure(figsize = (10,10))
                x = range(cur_iter//every)
                plt.plot(x,e_neg_std_total)
                plt.xlabel('Iteration / '+str(every))
                plt.ylabel('negative energy std')
                plt.grid()
                plt.savefig(args.save_dir + str(cur_iter)+"negative energy std")
                plt.close(0)


                #plot sample_method
                plt.figure(figsize = (10,10))
                x = range(cur_iter//every)
                plt.plot(x,sample_method)
                plt.xlabel('Iteration / '+str(every))
                plt.ylabel('')
                plt.grid()
                plt.savefig(args.save_dir + str(cur_iter)+"sample_method value")
                plt.close(0)

                ###################RHAT####################
                # en_list.append(en_rhat1)
                # en_list.append(en_rhat2)
                # en_list.append(en_rhat3)
                # en_list.append(en_rhat4)
                # all_en.append(en_list)

                # W = W_whitin_chain(all_en, args.n_steps, 0, num_chains)
                # B = B_between_chain(all_en, args.n_steps, 0, num_chains)

                # VAR = (((num_chains-1) / num_chains) * W + (1 / num_chains) * B)

                # R = torch.sqrt(VAR/W)
                # R_ave = sum(R,0)
                # R_ave /= 1000


                # from torchmetrics import IS, FID, KID
                # inception = IS().to(device, non_blocking=True)
                # fid = FID(feature=2048).to(device, non_blocking=True)

                # fid.update(x_p_d, real=True)
                # fid.update(x_q, real=False)
                # fid_val = fid.compute()
                # Fid_list.append(fid_val.item())

                # #plot FID
                # plt.figure(figsize = (10,10))
                # x = range(cur_iter/every)
                # plt.plot(x,Fid_list)
                # plt.xlabel('Iteration / '+str(every))
                # plt.ylabel('Fid')
                # plt.savefig(args.save_dir + str(cur_iter)+"FID")
                # plt.close(0)


                # #plot IS
                # inception.update(x_q)
                # inception_mean, inception_std = inception.compute()
                # Is_mean_list.append(inception_mean.item())
                # Is_std_list.append(inception_std.item())

                # plt.figure(figsize = (10,10))
                # x = range(cur_iter/every)
                # plt.plot(x,Is_mean_list)
                # plt.xlabel('Iteration / '+str(every))
                # plt.ylabel('IS mean')
                # plt.savefig(args.save_dir + str(cur_iter)+"IS mean")
                # plt.close(0)

                # plt.figure(figsize = (10,10))
                # x = range(cur_iter/every)
                # plt.plot(x,Is_std_list)
                # plt.xlabel('Iteration / '+str(every))
                # plt.ylabel('IS std')
                # plt.savefig(args.save_dir + str(cur_iter)+"IS std")
                # plt.close(0)


        if epoch % args.ckpt_every == 0:
            checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt', args, device)

        if epoch % args.eval_every == 0 and (args.p_y_given_x_weight > 0 or args.p_x_y_weight > 0):
            f.eval()
            with t.no_grad():
                # validation set
                correct, loss = eval_classification(f, dload_valid, device)
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, replay_buffer, "best_valid_ckpt.pt", args, device)
                # test set
                correct, loss = eval_classification(f, dload_test, device)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
            f.train()
        checkpoint(f, replay_buffer, "last_ckpt.pt", args, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--p_x_y_weight", type=float, default=0.)
    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"],
                        help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    parser.add_argument("--n_valid", type=int, default=5000)

    #sampling
    parser.add_argument("--sample", type=str, default='sgld')
    parser.add_argument("--rhat", type=int, default=0)
    parser.add_argument("--fid_every", type=int, default=100)
    parser.add_argument("--tem", type=int, default=1)
    parser.add_argument("--mala_variance_tem", type=int, default=1)
    parser.add_argument("--Lambda", type=float, default=1)
    #parser.add_argument("--rhat", type=int, default=0)

    # args.tem
    # args.mala_variance


    args = parser.parse_args()
    args.n_classes = 100 if args.dataset == "cifar100" else 10
    main(args)
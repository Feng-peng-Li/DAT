from __future__ import print_function
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import Bar, Logger, AverageMeter, accuracy
from utils_awp import TradesAWP
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from PIL import Image

from models.resnet import ResNet18
import models

import torchvision
import kornia.augmentation as K
from transform import get_transform,RandomCrop

import random
from torchvision.utils import save_image
from Generator import Generator_MLP


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--arch', type=str, default='ResNet18')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--JS_weight', default=2., type=int, metavar='N',
                    help='The weight of the JS divergence term')
parser.add_argument('--gpu_id', default=1, type=int,
                    help='gpu')
parser.add_argument('--beta', default=15.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')

parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--in_ch', default=100, type=int,
                    help='channel of input noise')
parser.add_argument('--mix_beta', type=float, default=1., metavar='mix_beta',
                    help='mix_rate')
parser.add_argument('--mix_beta1', type=float, default=0.2, metavar='mix_beta1',
                    help='mix_rate')
parser.add_argument('--mix_num', default=50, type=int,
                    help='perturbation')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                    help='retrain from which epoch')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--data-path', type=str, default='./data',
                    help='where is the dataset CIFAR-10')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'],
                    help='The threat model')
parser.add_argument('--train_budget', default='high', type=str, choices=['low', 'high'],
                    help='The compute budget for training. High budget would mean larger number of atatck iterations')
parser.add_argument('--epsilon', default=8, type=float,
                    help='perturbation')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--LS', type=int, default=0, metavar='S',
                    help='make 1 is want to use Label Smoothing. DAJAT uses LS only for CIFAR100 dataset')
parser.add_argument('--model-dir', default='./model-cifar-ResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--resume-model', default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim', default='', type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--resume-model1', default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim1', default='', type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--resume-model2', default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim2', default='', type=str,
                    help='directory of optimizer for retraining')

parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--num_auto', default=1, type=int, metavar='N',
                    help='Number of autoaugments to use for training')

parser.add_argument('--awp-gamma', default=0.005, type=float,
                    help='whether or not to add parametric noise')
parser.add_argument('--awp-warmup', default=10000, type=int,
                    help='We could apply AWP after some epochs for accelerating.')
parser.add_argument('--use_defaults', type=str, default='NONE',
                    choices=['NONE', 'CIFAR10_RN18', 'CIFAR10_WRN', 'CIFAR100_WRN', 'CIFAR100_RN18'],
                    help='Use None is want to use the hyperparamters passed in the python training command else use the desired set of default hyperparameters')

args = parser.parse_args()
# if args.use_defaults != 'NONE':
#     args = use_default(args.use_defaults)
print(args)

epsilon = args.epsilon / 255
args.epsilon = epsilon
if args.awp_gamma <= 0.0:
    args.awp_warmup = np.infty
if args.data == 'CIFAR100':
    NUM_CLASSES = 100

else:
    NUM_CLASSES = 10


# settings
model_dir = args.model_dir + '_' + str(args.gpu_id) + '_' + str(args.mix_num)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpu_id)
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


class Get_Dataset_C10(torchvision.datasets.CIFAR10):

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        image = self.transform[0](img)
        img_o = self.transform[1](img)
        return image, img_o, target

class Get_Dataset_C100(torchvision.datasets.CIFAR100):

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        image = Image.fromarray(img)
        image_clean = self.transform[0](image)
        image_auto1 = self.transform[1](image)

        return image_clean, image_auto1, target

transforms_train,transforms_test=get_transform()


if args.data == 'CIFAR10':
    trainset = Get_Dataset_C10(root=args.data_path, train=True, transform=transforms_train,
                               download=True)
elif args.data == 'CIFAR100':
    trainset = Get_Dataset_C100(root=args.data_path, train=True, transform=transforms_train,
                                download=True)

testset = getattr(datasets, args.data)(root=args.data_path, train=False, download=True, transform=transform_test)
valset = getattr(datasets, args.data)(root=args.data_path, train=True, download=True, transform=transform_test)

train_size = 49000
valid_size = 1000
test_size  = 10000
train_indices = list(range(50000))
val_indices = []
count = np.zeros(100)
for index in range(len(trainset)):
    _,_,_, target = trainset[index]
    if(np.all(count==10)):
        break
    if(count[target]<10):
        count[target] += 1
        val_indices.append(index)
        train_indices.remove(index)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,sampler=SubsetRandomSampler(val_indices), **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def get_fft(x):
    fft_im = torch.fft.fftn(x, dim=(-2, -1))  # bx3xhxw
    fft_amp = torch.abs(fft_im)
    fft_pha = torch.angle(fft_im)
    fft_amp = torch.fft.fftshift(fft_amp, dim=(-2, -1))
    return fft_amp, fft_pha


def inverse_fft(fft_amp, fft_pha):
    fft_amp = torch.fft.ifftshift(fft_amp, dim=(-2, -1))
    img_ifft = fft_amp * torch.exp(1j * fft_pha)
    img_ifft = torch.fft.ifftn(img_ifft, dim=(-2, -1))
    img = torch.real(img_ifft).float()

    img = torch.clip(img, 0., 1.0)
    return img


def perturb_input(model,
                  x_natural,
                  target,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf', batch_norm='base'):
    model.eval()

    if distance == 'l_inf':
        # x_adv = x_natural.detach() +((4.0 / 255.0) * torch.sign(torch.tensor([0.5]).to(device) - torch.rand_like(x_natural)).to(device) )
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv=torch.clamp(x_adv,0.,1.)
        bs=x_adv.size(0)
        # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logit_adv=model(x_adv, batch_norm)
                loss_kl =F.cross_entropy(logit_adv,target,label_smoothing=args.alpha) +args.beta*F.kl_div(F.log_softmax(logit_adv, dim=1),
                                   F.softmax(model(x_natural,batch_norm), dim=1),
                                   reduction='batchmean')
                # loss_kl=F.cross_entropy(logit_adv,target)+args.beta*torch.norm(model(x_natural,batch_norm) - logit_adv, 'nuc')/bs
                # loss_kl=symmkl(model(x_adv, batch_norm),model(x_natural, batch_norm))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def one_step(model,c_criterion, args, x_base, x_adv_base, x_auto1, x_adv_auto1, x_auto2, x_adv_auto2, target, epoch, awp_adversary,
             train=True):


    bs=x_base.size(0)
    logits_base,f_base = model(x_base,'base',train=True)
    logits_adv_base,f_adv_base = model(x_adv_base,'base',train=True)

    p_base = F.softmax(logits_base, dim=1)
    p_adv_base = F.softmax(logits_adv_base, dim=1)


    if args.LS == 1:
        criterion = LabelSmoothingLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.alpha)

    loss_robust_base = F.kl_div(F.log_softmax(logits_adv_base, dim=1),
                                F.softmax(logits_base, dim=1),
                                reduction='batchmean')
    # loss_robust_base = torch.norm(logits_base - logits_adv_base, 'nuc') / bs


    loss_natural_base = criterion(logits_base, target)
    loss_base = loss_natural_base + args.beta * loss_robust_base

    if args.num_auto >= 1:
        logits_auto1, feature1 = model(x_auto1,'auto', train=True)

        logits_adv_auto1,f_auto1 = model(x_adv_auto1,'auto', train=True)
        loss_robust_auto1 = F.kl_div(F.log_softmax(logits_adv_auto1, dim=1),
                                     F.softmax(logits_auto1, dim=1),
                                     reduction='batchmean')
        # loss_robust_auto1=torch.norm(logits_auto1 - logits_adv_auto1, 'nuc')/bs

        p_auto1 = F.softmax(logits_auto1, dim=1)
        p_adv_auto1 = F.softmax(logits_adv_auto1, dim=1)

        loss_natural_auto1 = criterion(logits_auto1, target)
        loss_auto1 = loss_natural_auto1 + args.beta * loss_robust_auto1

        if args.num_auto == 1:

            p_mixture = torch.clamp((p_base + p_auto1) / 2., 1e-7, 1).log()


            loss_JS = (F.kl_div(p_mixture, p_base, reduction='batchmean') + F.kl_div(p_mixture, p_auto1,
                                                                                     reduction='batchmean')) / 2

            loss=(loss_base+loss_auto1)/2.+args.JS_weight * (loss_JS)
            # loss = (loss_base+loss_auto1)/2. + args.JS_weight * loss_JS
                   # +args.beta2*loss_adv_JS

    if args.num_auto >= 2:
        logits_auto2, feature2 = model(x_auto2, 'auto', train=True)
        logits_adv_auto2,f_auto2 = model(x_adv_auto2, 'auto', train=True)
        loss_robust_auto2 = F.kl_div(F.log_softmax(logits_adv_auto2, dim=1),
                                     F.softmax(logits_auto2, dim=1),
                                     reduction='batchmean')



        p_auto2 = F.softmax(logits_auto2, dim=1)

        loss_natural_auto2 = criterion(logits_auto2, target)
        loss_auto2 = loss_natural_auto2 + args.beta2 * loss_robust_auto2

        p_mixture = torch.clamp((p_base + p_auto1 + p_auto2) / 3., 1e-7, 1).log()

        loss_JS = (F.kl_div(p_mixture, p_base, reduction='batchmean') + F.kl_div(p_mixture, p_auto1,
                                                                                 reduction='batchmean') + F.kl_div(
            p_mixture, p_auto2, reduction='batchmean')) / 3
        # loss_JS = (symmkl(p_mixture, p_auto1) + symmkl(p_mixture, p_base)+ symmkl(p_mixture, p_auto2)) / 3
        loss = (loss_base + loss_auto1 + loss_auto2) / 3 + args.JS_weight * (loss_JS)
    if train:
        return loss, logits_adv_base
    else:
        return -loss


def adv_train(model,c_criterion,train_loader, optimizer, a_G, optim_G, a_G2, optim_G2, epoch, awp_adversary, start_wa, tau_list,exp_avgs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    print('epoch: {}'.format(epoch))
    bar = Bar('Processing', max=len(train_loader))

    # varepsilon = args.epsilon * ((epoch) / args.epochs)
    varepsilon = 8 / 255.
    step_size = 2 / 255.
    iters_attack = 5
    # Crop=Randomcrop(0,11)
    K_aug = nn.Sequential(
        K.RandomHorizontalFlip(),
        K.RandomCrop((32,32),padding=3),)

    for batch_idx, (data_base, img_o, target) in enumerate(train_loader):
        x_base, img_o, target = data_base.to(device), img_o.to(device), target.to(device)
        bs = x_base.size(0)

        x_adv_base = perturb_input(model=model,
                                   x_natural=x_base,
                                   target=target,
                                   step_size=step_size,
                                   epsilon=varepsilon,
                                   perturb_steps=iters_attack,
                                   distance=args.norm, batch_norm='base')

        # x_amp, x_pha = get_fft(x_base.detach())
        na_amp,na_pha=get_fft(img_o)

        model.eval()

        # latent=model(x_adv_base.detach())
        latent_na = model(img_o.detach())



        z = torch.randn((bs, args.in_ch)).to(device)

        if args.num_auto >= 1:
            beta1 = np.random.uniform(0, args.mix_beta, (bs, 1))
            beta1 = torch.from_numpy(beta1).to(device)
            beta1 = beta1.view(bs, 1, 1, 1)


            amp1_na=a_G(z,latent_na.detach())
            x_amp1_na= beta1 * amp1_na + (1 - beta1) * na_amp
            x_auto1=inverse_fft(x_amp1_na,na_pha)
            x_auto1=K_aug(x_auto1)

            x_auto2 = x_auto1.clone().detach()

        # if args.num_auto >= 2:
        #     amp2 = a_G2(z,latent.detach())
        #     beta2 = np.random.uniform(0, args.mix_beta, (bs, 1))
        #     beta2 = torch.from_numpy(beta2).to(device)
        #     beta2 = beta2.view(bs, 1, 1, 1)
        #     x_amp2 = beta2 * amp2 + (1 - beta2) * x_amp
        #     x_auto2 = inverse_fft(x_amp2, x_pha)
        #     x_auto2 = K_aug(x_auto2)

        if args.num_auto >= 1:
            x_adv_auto1 = perturb_input(model=model,
                                        x_natural=x_auto1,
                                        target=target,
                                        step_size=step_size,
                                        epsilon=varepsilon,
                                        perturb_steps=iters_attack,
                                        distance=args.norm, batch_norm='auto')
        
            x_adv_auto2 = x_adv_auto1.clone().detach()
        # if args.num_auto >= 2:
        #     x_adv_auto2 = perturb_input(model=model,
        #                                 x_natural=x_auto2,
        #                                 step_size=step_size,
        #                                 epsilon=varepsilon,
        #                                 perturb_steps=iters_attack,
        #                                 distance=args.norm, batch_norm='auto')

        model.train()
        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv_base,
                                         inputs_clean=x_base, auto1=x_auto1.detach(), auto1_adv=x_adv_auto1.detach(),
                                         targets=target,
                                         args=args)
            awp_adversary.perturb(awp)
        loss, logits_adv_base = one_step(model,c_criterion,args, x_base, x_adv_base, x_auto1.detach(), x_adv_auto1,
                                         x_auto2.detach(), x_adv_auto2, target, epoch, awp_adversary)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        g_loss = one_step(model,c_criterion,args, x_base, x_adv_base, x_auto1, x_adv_auto1, x_auto2, x_adv_auto2, target, epoch,
                          awp_adversary, train=False)

        if args.num_auto >= 1:
                optim_G.zero_grad()

        # if args.num_auto >= 2:
        #         optim_G2.zero_grad()
        g_loss.backward()
        if args.num_auto >= 1:
                # torch.nn.utils.clip_grad_norm_(a_G.parameters(), 2.0)
                optim_G.step()
        # if args.num_auto >= 2:
        #         optim_G2.step()
        # update generator

        prec1, prec5 = accuracy(logits_adv_base, target, topk=(1, 5))
        losses.update(loss.item(), x_base.size(0))
        top1.update(prec1.item(), x_base.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg, exp_avgs



def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


upper_limit, lower_limit = 1, 0


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, soft_label=None, epoch=0, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(X + delta), y_a, y_b, lam)
            else:

                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(X + delta), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(X + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def test(model, test_loader, criterion):
    global best_acc
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(test_loader))

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        x_perturb = attack_pgd(model, inputs, targets, 8 / 255., 2 / 255., 20, 1, args.norm)
        x_perturb.detach()
        x_adv = torch.clamp(inputs + x_perturb[:inputs.size(0)], min=0., max=1.0)
        outputs = model(x_adv)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total: {total:}| ETA: {eta:}| Loss:{loss:.4f}| top1: {top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(test_loader),
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 110:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    ################### Change here to WideResNet34 if you want to train on WRN-34-10 #############################
    model = getattr(models, args.arch)(num_classes=NUM_CLASSES).to(device)

    a_G = Generator_MLP(args.in_ch, 3, 32, 32,NUM_CLASSES).to(device)

    a_G2 = Generator_MLP(args.in_ch, 3, 32, 32,NUM_CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optim_G = optim.SGD(a_G.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optim_G2 = optim.SGD(a_G2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optim_G = optim.Adam(a_G.parameters(), lr=0.0002,betas=(0.5,0.999))
    # optim_G2 = optim.Adam(a_G2.parameters(), lr=0.0002,betas=(0.5,0.999))
    ################### Change here to WideResNet34 if you want to train on WRN-34-10 #############################
    proxy = getattr(models, args.arch)(num_classes=NUM_CLASSES).to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)
    class_center=torch.zeros([len(trainset),NUM_CLASSES]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.alpha)
    logger = Logger(os.path.join(model_dir, 'log.txt'), title=args.arch)
    logger.set_names(['Learning Rate', 'Nat Val Loss', 'Nat Val Acc.'])
    # torch.load(class_center, 'centers.t')
    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))
        a_G.load_state_dict(torch.load(args.resume_model1, map_location=device))
        # a_G2.load_state_dict(torch.load(args.resume_model2, map_location=device))
    if args.resume_optim:
        optimizer.load_state_dict(torch.load(args.resume_optim, map_location=device))
        optim_G.load_state_dict(torch.load(args.resume_optim1, map_location=device))
        # optim_G2.load_state_dict(torch.load(args.resume_optim2, map_location=device))
    start_wa = [(150 * args.epochs) // 200]
    tau_list = [0.9996]
    exp_avgs = []
    model_tau = getattr(models, args.arch)(num_classes=NUM_CLASSES).to(device)
    exp_avgs.append(model_tau.state_dict())
    num = int(args.mix_num)
    best_val_acc=0.
    best_test_acc=0.
    for epoch in range(args.start_epoch, args.epochs + 1):

        lr = adjust_learning_rate(optimizer, epoch)

        adv_loss, adv_acc, exp_avgs = adv_train(model,class_center,train_loader, optimizer, a_G, optim_G, a_G2, optim_G2, epoch,
                                                awp_adversary, start_wa, tau_list,
                                                exp_avgs)
        print('================================================================')
        val_loss, val_acc = test(model, val_loader, criterion)
        print('================================================================')

        logger.append([lr, val_loss, val_acc])
        if val_acc>=best_val_acc:
            best_val_acc=val_acc
            test_loss, test_acc = test(model, test_loader, criterion)
            logger.append([lr, test_loss, test_acc])
            if best_test_acc<=test_acc:
                best_test_acc=test_acc

            torch.save(model.state_dict(),
                           os.path.join(model_dir, 'ours-model-epoch{}.pt'.format(epoch)))
            torch.save(a_G.state_dict(),
                           os.path.join(model_dir, 'ours-Gmodel-epoch{}.pt'.format(epoch)))
            # torch.save(a_G2.state_dict(),
            #                os.path.join(model_dir, 'ours-Gmodel2-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                           os.path.join(model_dir, 'ours-opt-checkpoint_epoch{}.tar'.format(epoch)))
            torch.save(optim_G.state_dict(),
                           os.path.join(model_dir, 'ours-optG-checkpoint_epoch{}.tar'.format(epoch)))
            # torch.save(optim_G2.state_dict(),
            #                os.path.join(model_dir, 'ours-optG2-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()

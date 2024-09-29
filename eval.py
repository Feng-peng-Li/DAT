import os
import argparse
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import models
import sys
import torch.nn.functional as F

sys.path.insert(0, '..')


def replicate_input(x):
    return x.detach().clone()


def to_one_hot(y, num_classes=10):
    """
    Take a batch of label y with n dims and convert it to
    1-hot representation with n+1 dims.
    Link: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24
    """
    y = replicate_input(y).view(-1, 1)
    y_one_hot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    return y_one_hot


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


def normalize(X):
    return (X - mu) / std


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


upper_limit, lower_limit = 1, 0


class CarliniWagnerLoss(nn.Module):
    """
    Carlini-Wagner Loss: objective function #6.
    Paper: https://arxiv.org/pdf/1608.04644.pdf
    """

    def __init__(self, kappa=10.0):
        super(CarliniWagnerLoss, self).__init__()
        self.kappa = kappa

    def forward(self, input, target):
        """
        :param input: pre-softmax/logits.
        :param target: true labels.
        :return: CW loss value.
        """
        num_classes = input.size(1)
        label_mask = to_one_hot(target, num_classes=num_classes).float()
        correct_logit = torch.sum(label_mask * input, dim=1)
        wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + self.kappa).sum()
        return loss


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False, loss_type='CrossEntropyLoss',
               is_random=True):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if is_random:
            if norm == "Linf":
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
            if loss_type == 'CrossEntropyLoss':
                loss = F.cross_entropy(output, y)
            elif loss_type == 'CWLoss':
                loss = CarliniWagnerLoss()(output, y)
            else:
                raise ValueError('Please use valid losses.')
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "Linf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ResNet18',
                        choices=['WideResNet28', 'WideResNet34', 'ResNet18'])
    parser.add_argument('--attack', type=str, default='PGD',
                        choices=['WideResNet28', 'WideResNet34', 'ResNet18'])
    parser.add_argument('--gpu_id', type=int, default=4)
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--no-random', action='store_false')
    parser.add_argument('--checkpoint', type=str,
                        default='')
    parser.add_argument('--data', type=str, default='CIFAR100', choices=['CIFAR10', 'CIFAR100'],
                        help='Which dataset the eval is on')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--preprocess', type=str, default='01',
                        choices=['meanstd', '01', '+-1'], help='The preprocess for data')
    parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'])
    parser.add_argument('--epsilon', type=float, default=8. / 255.)

    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--individual', default=False, action='store_true')
    parser.add_argument('--save_dir', type=str, default='./adv_inputs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--log_path', type=str, default='./log.txt')
    parser.add_argument('--version', type=str, default='standard')

    args = parser.parse_args()
    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu_id)
    num_classes = int(args.data[5:])
    eval_type = args.attack
    if args.preprocess == 'meanstd':
        if args.data == 'CIFAR10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif args.data == 'CIFAR100':
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    elif args.preprocess == '01':
        mean = (0, 0, 0)
        std = (1, 1, 1)
    elif args.preprocess == '+-1':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError('Please use valid parameters for normalization.')

    # model = ResNet18()
    if args.arch == 'WideResNet34':
        net =getattr(models, args.arch)(depth=34, num_classes=num_classes, widen_factor=10).to(device)
    elif args.arch == 'WideResNet28':
        net = getattr(models, args.arch)(depth=28, num_classes=num_classes, widen_factor=10).to(device)
    elif args.arch == 'ResNet18':
        net = getattr(models, args.arch)(num_classes=num_classes).to(device)
    else:
        raise ValueError('Please use choose correct architectures.')

    ckpt =torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt)

    model = nn.Sequential(Normalize(mean=mean, std=std), net)

    model.to(device)
    model.eval()

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    item = getattr(datasets, args.data)(root=args.data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load attack
    if eval_type == 'AA':
        from autoattack import AutoAttack

        adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path)

        l = [x for (x, y) in test_loader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0)

        # cheap version
        # example of custom version
        if args.version == 'custom':
            adversary.attacks_to_run = ['apgd-ce', 'fab']
            adversary.apgd.n_restarts = 2
            adversary.fab.n_restarts = 2

        # run attack and save images
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                                                             bs=args.batch_size)

            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                args.save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon))
    else:
        model.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss_fgsm = 0
        test_robust_acc_fgsm = 0
        test_robust_loss_pgd20 = 0
        test_robust_acc_pgd20 = 0
        test_robust_loss_pgd100 = 0
        test_robust_acc_pgd100 = 0
        test_robust_loss_cw = 0
        test_robust_acc_cw = 0
        test_n = 0
        criterion = nn.CrossEntropyLoss()
        start_time = time.time()
        for i, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)

            # Random initialization
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta_fgsm = attack_pgd(model, X, y, args.epsilon, 16 / 255., 1, args.restarts, args.norm,
                                        early_stop=False,
                                        loss_type='CrossEntropyLoss', is_random=args.no_random)
                delta_pgd20 = attack_pgd(model, X, y, args.epsilon, 2 / 255., 20, args.restarts, args.norm,
                                         early_stop=False,
                                         loss_type='CrossEntropyLoss', is_random=args.no_random)
                delta_pgd100 = attack_pgd(model, X, y, args.epsilon, 2 / 255., 100, args.restarts, args.norm,
                                          early_stop=False,
                                          loss_type='CrossEntropyLoss', is_random=args.no_random)
                delta_cw = attack_pgd(model, X, y, args.epsilon, 2 / 255., 100, args.restarts, args.norm,
                                      early_stop=False,
                                      loss_type='CWLoss', is_random=args.no_random)
            delta_fgsm = delta_fgsm.detach()
            delta_pgd20 = delta_pgd20.detach()
            # delta_pgd100 = delta_pgd100.detach()
            delta_cw = delta_cw.detach()

            robust_output_fgsm = model(
                torch.clamp(X + delta_fgsm[:X.size(0)], min=lower_limit, max=upper_limit))
            robust_loss_fgsm = criterion(robust_output_fgsm, y)

            robust_output_pgd20 = model(
                torch.clamp(X + delta_pgd20[:X.size(0)], min=lower_limit, max=upper_limit))
            robust_loss_pgd20 = criterion(robust_output_pgd20, y)

            robust_output_pgd100 = model(
                torch.clamp(X + delta_pgd100[:X.size(0)], min=lower_limit, max=upper_limit))
            robust_loss_pgd100 = criterion(robust_output_pgd100, y)

            robust_output_cw = model(torch.clamp(X + delta_cw[:X.size(0)], min=lower_limit, max=upper_limit))
            robust_loss_cw = criterion(robust_output_cw, y)

            output = model(X)
            loss = criterion(output, y)

            test_robust_loss_fgsm += robust_loss_fgsm.item() * y.size(0)
            test_robust_acc_fgsm += (robust_output_fgsm.max(1)[1] == y).sum().item()
            test_robust_loss_pgd20 += robust_loss_pgd20.item() * y.size(0)
            test_robust_acc_pgd20 += (robust_output_pgd20.max(1)[1] == y).sum().item()
            test_robust_loss_pgd100 += robust_loss_pgd100.item() * y.size(0)
            test_robust_acc_pgd100 += (robust_output_pgd100.max(1)[1] == y).sum().item()
            test_robust_loss_cw += robust_loss_cw.item() * y.size(0)
            test_robust_acc_cw += (robust_output_cw.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)
        test_time = time.time() - start_time
        print('{}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}'.format(
            test_time, test_robust_loss_fgsm / test_n, test_robust_acc_fgsm / test_n,
                       test_robust_loss_pgd20 / test_n, test_robust_acc_pgd20 / test_n,
                       test_robust_loss_pgd100 / test_n, test_robust_acc_pgd100 / test_n,
                       test_robust_loss_cw / test_n, test_robust_acc_cw / test_n,
                       test_loss / test_n, test_acc / test_n))


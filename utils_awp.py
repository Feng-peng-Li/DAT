import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class TradesAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, inputs_clean,auto1,auto1_adv, targets, args):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        logits_base=self.proxy(inputs_clean)
        p_base=F.softmax(logits_base,dim=1)
        loss_natural = F.cross_entropy(logits_base, targets,label_smoothing=args.alpha)
        loss_robust = F.kl_div(F.log_softmax(self.proxy(inputs_adv), dim=1),
                               p_base,
                               reduction='batchmean')
        loss_base=loss_natural + args.beta * loss_robust

        logits_auto1=self.proxy(auto1, 'auto')
        p_auto1=F.softmax(logits_auto1,dim=1)
        loss_auto1 = F.cross_entropy(logits_auto1, targets, label_smoothing=args.alpha)
        loss_robust_auto1 = F.kl_div(F.log_softmax(self.proxy(auto1_adv, 'auto'), dim=1),
                               p_auto1,
                               reduction='batchmean')
        loss_a = loss_auto1 + args.beta * loss_robust_auto1

        p_mixture = torch.clamp((p_base + p_auto1) / 2., 1e-7, 1).log()


        loss_JS = (F.kl_div(p_mixture, p_base, reduction='batchmean') + F.kl_div(p_mixture, p_auto1,
                                                                                 reduction='batchmean')) / 2

        loss_toal=(loss_a+loss_base)/2.+args.JS_weight*loss_JS


        loss = - 1.0 * (loss_toal)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)





import copy

import torch
import math


def needs_mask(name):
    return name.endswith('weight')


def param_prune(model, mask, prune_proportion):
    """
        prune a proportion of parameters
    """
    new_mask = copy.deepcopy(mask)
    n_prune_dict = {}

    first_layer = True
    for mname in mask:
        if not needs_mask(mname):
            continue
        if first_layer:
            first_layer = False
            continue
        m = mask[mname]
        n_non_zeros = torch.sum(m)
        n_prune = math.ceil(prune_proportion * n_non_zeros)
        n_prune_dict[mname] = n_prune

        param = model.state_dict()[mname]
        temp_weights = torch.where(m > 0, torch.abs(param.data), 100000 * torch.ones_like(param.data))
        x, idx = torch.sort(temp_weights.view(-1))
        new_mask[mname].view(-1)[idx[:n_prune]] = 0
    return new_mask, n_prune_dict


def param_prune_to_sparsity(model, prune_sparsity):
    """
        prune model to target sparsity
    """
    new_mask = {}
    first_layer = True
    for pname, param in model.named_parameters():
        new_mask[pname] = torch.ones_like(param, dtype=torch.int)
        if not needs_mask(pname):
            continue
        if first_layer:
            first_layer = False
            continue
        n_param = param.numel()
        n_prune = math.ceil(prune_sparsity * n_param)
        temp_weights = torch.where(new_mask[pname] > 0, torch.abs(param.data), 100000 * torch.ones_like(param.data))
        x, idx = torch.sort(temp_weights.view(-1))
        new_mask[pname].view(-1)[idx[:n_prune]] = 0
    return new_mask


def param_regrow(model, mask, n_prune_dict):
    new_mask = copy.deepcopy(mask)
    first_layer = True

    for pname, param in model.named_parameters():
        if not needs_mask(pname):
            continue
        if first_layer:
            first_layer = False
            continue
        m = mask[pname]
        temp = torch.where(m == 0, torch.abs(param.grad), -100000 * torch.ones_like(param.grad))
        sort_temp, idx = torch.sort(temp.view(-1), descending=True)
        new_mask[pname].view(-1)[idx[:n_prune_dict[pname]]] = 1

    return new_mask

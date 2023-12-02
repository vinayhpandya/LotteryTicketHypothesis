import math

import torch
import torch.nn as nn
from torch.nn.utils.prune import l1_unstructured, random_unstructured


class MLP(nn.Module):
    """A Simple Multi Layer Perceptron

    Description:
        This simple neural network will create a neuron
        and will be pruned to see if the network still
        performs good with 90% of the weights removed

    Parameters:
        n_features: int
            number of features of the input image (28*28)
        hidden_layers: tuple
            Tuple representing sizes of input layers inside the model
        n_target: int
            class labels, in this case 10
    """

    def __init__(self, n_features, hidden_layer_sizes, n_targets):
        super().__init__()
        layer_sizes = (n_features,) + hidden_layer_sizes + (n_targets,)
        layer_list = []
        for i in range(len(layer_sizes) - 1):
            layer_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x):
        """Forward pass through the MLP Model

        Args:
            x ( torch.Tensor): An input tensor
        """
        n_layers = len(self.module_list)
        for i, layer in enumerate(self.module_list):
            x = layer(x)
            if i < n_layers - 1:
                x = nn.functional.relu(x)
            return x


def prune_linear(linear, ratio=0.3, method="l1"):
    if method == "l1":
        prune_func = l1_unstructured
    elif method == "random":
        prune_func = random_unstructured
    prune_func(linear, "weight", ratio)
    prune_func(linear, "bias", ratio)


def prune_mlp(mlp, prune_ratio, method="l1"):
    if isinstance(prune_ratio, float):
        prune_ratios = [prune_ratio] * len(mlp.module_list)
    else:
        prune_ratios = prune_ratio
    for prune_ratio, linear in zip(prune_ratios, mlp.module_list):
        prune_linear(linear, prune_ratio, method=method)


def check_pruned_linear(linear):
    params = {params for params, _ in linear.named_parameters()}
    expected_params = {"weight_orig", "bias_orig"}

    return params == expected_params


def reinit_layer(linear):
    is_pruned = check_pruned_linear(linear)
    if is_pruned:
        weight = linear.weight_orig
        bias = linear.bias_orig
    else:
        weight = linear.weight
        bias = linear.weight

    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)


"""
This step is important because this will make sure that the subnetworks sampled are
not due to some optimization trajectory but because of networks architecture or 
weight distribution
"""


def reinit_mlp(mlp):
    for linear in mlp.module_list:
        reinit_layer(linear)


def copy_weights_linear(linear_unpruned, linear_pruned):
    """Copy weights from an unpruned model to a pruned model.

    Modifies `linear_pruned` in place.

    Parameters
    ----------
    linear_unpruned : nn.Linear
        Linear model with a bias that was not pruned.

    linear_pruned : nn.Linear
        Linear model with a bias that was pruned.
    """
    assert check_pruned_linear(linear_pruned)
    assert not check_pruned_linear(linear_unpruned)

    with torch.no_grad():
        linear_pruned.weight_orig.copy_(linear_unpruned.weight)
        linear_pruned.bias_orig.copy_(linear_unpruned.bias)


def copy_weights_mlp(mlp_unpruned, mlp_pruned):
    """Copy weights of an unpruned network to a pruned network.

    Modifies `mlp_pruned` in place.

    Parameters
    ----------
    mlp_unpruned : MLP
        MLP model that was not pruned.

    mlp_pruned : MLP
        MLP model that was pruned.
    """
    zipped = zip(mlp_unpruned.module_list, mlp_pruned.module_list)

    for linear_unpruned, linear_pruned in zipped:
        copy_weights_linear(linear_unpruned, linear_pruned)


def compute_stats(mlp):
    stats = {}
    total_params = 0
    total_pruned_params = 0
    for layer_ix, linear in enumerate(mlp.module_list):
        assert check_pruned_linear(linear)

        weight_mask = linear.weight_mask
        bias_mask = linear.bias_mask

        params = weight_mask.numel() + bias_mask.numel()
        pruned_params = (weight_mask == 0).sum() + (bias_mask == 0).sum()

        total_params += params
        total_pruned_params += pruned_params

        stats[f"layer{layer_ix}_total_params"] = params
        stats[f"layer{layer_ix}_pruned_params"] = pruned_params
        stats[f"layer{layer_ix}_actual_prune_ratio"] = pruned_params / params

    stats["total_params"] = total_params
    stats["total_pruned_params"] = total_pruned_params
    stats["actual_prune_ratio"] = total_pruned_params / total_params

    return stats

"""
Local compatibility layer for learn2learn functionality.
This module provides minimal implementations of BaseLearner, clone_module, and update_module
to avoid the need to install learn2learn on Windows.
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class BaseLearner(nn.Module):
    """
    A simple base class for meta-learners.
    Based on learn2learn's BaseLearner implementation.
    """

    def __init__(self):
        super(BaseLearner, self).__init__()

    def __getattr__(self, attr):
        try:
            return super(BaseLearner, self).__getattr__(attr)
        except AttributeError:
            return getattr(self.__dict__['_modules']['module'], attr)


def clone_module(module, memo=None):
    """
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().

    This allows the cloned module to properly track gradients.
    
    Based on learn2learn's clone_module implementation.
    """
    if memo is None:
        memo = {}

    # First, create a copy of the module without its children
    if hasattr(module, '_parameters'):
        for param_key in module._parameters:
            param = module._parameters[param_key]
            if param is not None:
                cloned_param = param.clone()
                if param.requires_grad:
                    cloned_param.requires_grad_()
                memo[id(param)] = cloned_param

    if hasattr(module, '_buffers'):
        for buffer_key in module._buffers:
            buff = module._buffers[buffer_key]
            if buff is not None and id(buff) not in memo:
                memo[id(buff)] = buff.clone()

    # Clone the module
    if hasattr(module, '__deepcopy__'):
        clone = module.__deepcopy__(memo)
    else:
        clone = module.__class__.__new__(module.__class__)
        clone.__dict__ = module.__dict__.copy()

    # Re-assign parameters (pointing to cloned versions)
    if hasattr(clone, '_parameters'):
        for param_key in clone._parameters:
            if clone._parameters[param_key] is not None:
                param = module._parameters[param_key]
                cloned_param = memo.get(id(param), param.clone())
                if param.requires_grad:
                    cloned_param.requires_grad_()
                clone._parameters[param_key] = cloned_param

    # Re-assign buffers (pointing to cloned versions)
    if hasattr(clone, '_buffers'):
        for buffer_key in clone._buffers:
            if clone._buffers[buffer_key] is not None:
                buff = module._buffers[buffer_key]
                clone._buffers[buffer_key] = memo.get(id(buff), buff.clone())

    # Re-assign children (recursively clone)
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            if clone._modules[module_key] is not None:
                clone._modules[module_key] = clone_module(
                    module._modules[module_key], 
                    memo=memo
                )

    return clone


def update_module(module, updates=None, memo=None):
    """
    Updates the parameters of a module in-place, in a way that preserves differentiability.

    The parameters of the module are swapped with their update values, according to the
    .update attribute of each parameter.
    
    Based on learn2learn's update_module implementation.
    """
    if memo is None:
        memo = {}

    if hasattr(module, '_parameters'):
        for param_key in module._parameters:
            param = module._parameters[param_key]
            if param is not None:
                if id(param) in memo:
                    module._parameters[param_key] = memo[id(param)]
                else:
                    if hasattr(param, 'update') and param.update is not None:
                        updated = param + param.update
                        memo[id(param)] = updated
                        module._parameters[param_key] = updated

    if hasattr(module, '_modules'):
        for module_key in module._modules:
            module._modules[module_key] = update_module(
                module._modules[module_key],
                updates=updates,
                memo=memo,
            )

    return module

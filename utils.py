import math
import os
import torch
from torch.nn import Sequential


def init_trainable_parameters(model):
    def init_module(module, name, params):
        for param_name, init_func in params.items():
            if hasattr(module, param_name):
                weight = getattr(module, param_name).weight
                device = weight.device
                new_weight = init_func(weight.size()).to(device)
                weight.data = new_weight
                weight.requires_grad = True
                print(f"Initialized {param_name} for {name}")
    lora_params = {
        'lora_A': lambda size: torch.randn(size) * math.sqrt(2 / size[0]),
        'lora_B': lambda size: torch.randn(size) * 1e-6
    }
    adapter_params = {
        'adapter_w1': lambda size: torch.randn(size) * math.sqrt(2 / size[0]),
        'adapter_w2': lambda size: torch.randn(size) * 1e-6
    }
    for name, module in model.named_modules():
        if all(hasattr(module, param) for param in lora_params):
            print(name, module)
            init_module(module, name, lora_params)
        if all(hasattr(module, param) for param in adapter_params):
            print(name, module)
            init_module(module, name, adapter_params)

def convert_trainable_parameters(model, trainable_param_names):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += 1
        if any(substring in name for substring in trainable_param_names):
            param.requires_grad = True
            trainable_params += 1
        else:
            param.requires_grad = False
    print(
        f"Convert trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Print trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def print_filtered_model_size(model, param_names_to_exclude):
    excluded_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if not any(substring in name for substring in param_names_to_exclude):
            excluded_params += param.numel()
    print(
        f"Print filtered model size: {excluded_params:,} || all params: {all_param:,} || excluded%: {100 * excluded_params / all_param:.2f}"
    )
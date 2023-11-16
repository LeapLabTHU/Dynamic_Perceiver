from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator

'''
    Calculate the FLOPS of each exit without lazy prediction pruning
    Based on https://github.com/kalviny/MSDNet-PyTorch/blob/master/op_counter.py since this is what was used by the 2 baselines
    we're comparing to. Small modifications for our architecture.
'''

count_ops = 0
count_params = 0
cls_ops = []
cls_params = []
count_ops_at_exits = []

def get_num_gen(gen):
    return sum(1 for x in gen)


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])

### The input batch size should be 1 to call this function
def measure_layer(layer, x, num_classes):
    global count_ops, count_params, cls_ops, cls_params, count_ops_at_exits
    delta_ops = 0
    delta_params = 0
    multi_add = 1 # 1 is MAD, 2 is FLOPS
    type_name = get_layer_info(layer)
    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    ### ops_nonlinearity
    elif type_name in ['ReLU' ,'Hardswish']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    elif type_name in ['Hardsigmoid']:
        delta_ops = count_hardsigmoid(x, multi_add)
        delta_params = get_layer_param(layer)
    ### ops_pooling
    elif type_name in ['AvgPool2d', 'MaxPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)
    elif type_name in ['LayerNorm']:
        delta_ops = count_normalization(layer, x)
        delta_params = get_layer_param(layer)
    elif type_name in ['GELU', 'CustomGELU']:
        delta_ops = count_gelu(x, multi_add)
        delta_params = get_layer_param(layer)
    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel() if layer.bias is not None else 0
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['BatchNorm1d', 'BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout',
                       'MSDNFirstLayer', 'ConvBasic', 'ConvBN',
                       'ParallelModule', 'MSDNet', 'Sequential',
                       'MSDNLayer', 'ConvDownNormal', 'ConvNormal', 'ClassifierModule',
                       'Unfold', 'Identity', 'DropPath', 'LayerNorm']:
        delta_params = get_layer_param(layer)
    elif type_name in ['LearnableUncGate', 'IdentityGate']:
        delta_ops = layer.get_flops(num_classes)
        delta_params = get_layer_param(layer)

    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    if type_name == 'Linear':
        cls_ops.append(count_ops)
        cls_params.append(count_params)
        # print('---------------------')
        # print('FLOPs at linear: %f, Params: %f' % (count_ops, count_params))
    if type_name in ['LearnableUncGate', 'IdentityGate']:
        # print('---------------------')
        # print('FLOPs at gate: %f, Params: %f' % (count_ops, count_params))
        count_ops_at_exits.append(count_ops)
        cls_params.append(count_params)

    return

def measure_model_and_assign_cost_per_exit(model, H, W, num_classes = 10):
    global count_ops, count_params, cls_ops, cls_params, count_ops_at_exits
    model = model.to('cpu')
    model.eval()
    count_ops = 0
    count_params = 0
    model_blocks = list(model.named_children())
    model_blocks = list(map(lambda x: x[0], model_blocks))
    print(model_blocks)
    data = Variable(torch.zeros(1, 3, H, W)) # equivalent of batch of 1, 3 channels, height and width.
    # training_forward = model.forward
    # model.forward = model.forward_for_inference
    def should_measure(x):
        return is_leaf(x)

    def modify_forward(model):
        for child_name, child in model.named_children():
            if child_name in model_blocks:
                idx = model_blocks.index(child_name)
                print(f'{child_name}. {idx}/{len(model_blocks)}')
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x, num_classes)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward

                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)
    # model.forward = training_forward
    count_ops_at_exits.append(cls_ops[-1]) # need to add the last layer where there is no gate.
    # model.set_cost_per_exit(count_ops_at_exits)
    print(f"Model was successfully measured WITH OUR CODE. Total cost is {cls_ops[-1]}")
    return cls_ops, cls_params, count_ops_at_exits


# ADDED STUFF

# From https://github.com/google-research/electra/blob/master/flops_computation.py
def count_normalization(m: nn.modules.batchnorm._BatchNorm, x):
    x = x[0]
    flops = 5 * x.numel()
    return flops / 2

def count_gelu(x, multi_add):
    x_size = x.numel()
    # GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
    flops = x_size * 4 * multi_add # multi-add controls whether we're doing flops or mul-add
    return flops

def count_hardsigmoid(x, multi_add):
    return x.numel() * (1 + 1) * multi_add


# Helper function to measure the incurred cost from adding IMs and gates.

# def measure_arch_mul_add(net, args, device, transformer_layer_gating, img_size = 224, num_classes = 10):
#     # this is due to the way we detect exits by using a gate, this is so we can get a clear picture at every exit even
#     # before augmenting the backbone with IMs. We use identity gates which have no costs simply to detect where exits are.
#     net.set_learnable_gates(device,
#                             transformer_layer_gating,
#                             direct_exit_prob_param=True,
#                             gate_type=GateType.IDENTITY,
#                             proj_dim=int(args.proj_dim),
#                             num_proj=int(args.num_proj))
#     n_flops, n_params, n_flops_at_gates = measure_model_and_assign_cost_per_exit(net, img_size, img_size, num_classes=num_classes)
#     print(f"Before adding extra heads {args.arch}, {args.dataset}: {n_flops_at_gates}")
#     net.set_intermediate_heads(transformer_layer_gating)
#
#     n_flops, n_params, n_flops_at_gates = measure_model_and_assign_cost_per_exit(net, img_size, img_size, num_classes=num_classes)
#     print(f"After adding extra heads {args.arch}, {args.dataset}: {n_flops_at_gates}")
#
#     net.set_learnable_gates(device,
#                             transformer_layer_gating,
#                             direct_exit_prob_param=True,
#                             gate_type=GateType.IDENTITY if 'baseline' in args.arch else args.gate,
#                             proj_dim=int(args.proj_dim),
#                             num_proj=int(args.num_proj))
#
#     n_flops, n_params, n_flops_at_gates = measure_model_and_assign_cost_per_exit(net, img_size, img_size, num_classes=num_classes)
#     print(f"Fully augmented model {args.arch}, {args.dataset}: {n_flops_at_gates}")
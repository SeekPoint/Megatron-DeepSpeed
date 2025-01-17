# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from functools import partial

from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_world_size
from .mappings import copy_to_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region
from .mappings import scatter_to_tensor_model_parallel_region
from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility
from ..model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from megatron import get_args, mpu
import deepspeed.runtime.activation_checkpointing.checkpointing as ds_checkpointing


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}


def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, 'tensor_model_parallel') and
            param.tensor_model_parallel) or (
                get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    if ds_checkpointing.is_configured():
        global get_cuda_rng_tracker
        get_cuda_rng_tracker = ds_checkpointing.get_cuda_rng_tracker

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


def xavier_uniform_tensor_parallel_(tensor, gain=1., tp_degree=1):
    r"""
    This is a modified torch.nn.init.xavier_uniform_ with changes to support
    partitioned on the vocab size dim embedding with tensor parallel.

    Additional args:
    - tp_degree: degree of tensor parallel

    Note: the code assumes all partitions are equal in size
    """
    # receptive_field_size=1 as dim==2, so we don't need init._calculate_fan_in_and_fan_out
    fan_out, fan_in = tensor.shape
    fan_out *= tp_degree # tp splits on num_embeddings dim

    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return torch.nn.init._no_grad_uniform_(tensor, -a, a)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the defaults for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocabulary dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        args = get_args()

        # only the first stage embedding runs this class' forward. The head's embedding does its own
        # thing, so don't waste memory allocating LN weights.
        if mpu.is_pipeline_first_stage() and (args.use_bnb_optimizer or args.embed_layernorm):
            self.norm = LayerNorm(embedding_dim)

        if args.use_bnb_optimizer:
            # for BNB we ignore the passed init_method and use torch.nn.init.xavier_uniform_
            # modified to calculate std on the unpartitioned embedding
            init_method = partial(xavier_uniform_tensor_parallel_, tp_degree=self.tensor_model_parallel_size)

        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

        if args.use_bnb_optimizer:
            from bitsandbytes.optim import GlobalOptimManager
            GlobalOptimManager.get_instance().override_config(self.weight, 'optim_bits', 32)
            GlobalOptimManager.get_instance().register_parameters(self.weight)


    def forward(self, input_):
        if torch.any(input_ >= self.num_embeddings):
            raise ValueError(f"There is an input id in the input that is greater than the highest possible input id.\nInput: {input_}\nnum_embeddings: {self.num_embeddings}")

        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            # input_ is garanted to be in the range [0:self.vocab_end_index - self.vocab_start_index] thanks to the first check
            masked_input = input_

        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)

        if hasattr(self, 'norm'):
            output = self.norm(output)

        return output

class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """
    """
        列并行线性层.
        线性层定义为Y=XA+b. A沿着第二维进行并行，A = [A_1, ..., A_p]

        参数:
            input_size: 矩阵A的第一维度.
            output_size: 矩阵A的第二维度.
            bias: 若为true则添加bias.
            gather_output: 若为true，在输出上调用all-gather，使得Y对所有GPT都可访问.
            init_method: 随机初始化方法.
            stride: strided线性层.
    """
    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
	    # 获得张量并行组的world_size
        world_size = get_tensor_model_parallel_world_size()
        # 按照张量并行度(world_size)划分输出维度
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result 执行 XA^T+b
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            # 初始化张量. 若完整权重矩阵A为n*m，张量并行度为k，这里初始化的张量为n*(m/k)
            # 也就是张量并行组中的进程各自初始化持有的部分张量
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=args.params_dtype))
            # 使用init_method对权重矩阵self.weight进行随机初始化(CPU版)
            # self.master_weight在测试中使用，这里不需要关注
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            # 使用init_method对权重矩阵self.weight进行随机初始化(GPU版)
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=stride)

        if bias:
            # 实例化一个bias
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            # 将张量并行的相关信息追加至self.bias
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
	    # bias初始化为0
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        # Set up backprop all-reduce.
        # 前向传播时input_parallel就等于input_
        # 反向传播时在张量并在组内将梯度allreduce
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            # 收集张量并行组内的张量并进行拼接
            # 此时，output是非张量并行情况下前向传播的输出
            # 张量并行组中的进程都持有完全相同的output
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            # 此时，output是张量并行情况下的前向传播输出
            # 张量并行组中的进程持有不同的output
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """
    """
    行并行线性层.
    线性层的定义为Y = XA + b. x
    A沿着第一个维度并行，X沿着第二个维度并行. 即
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    参数:
        input_size: 矩阵A的第一维度.
        output_size: 矩阵A的第二维度.
        bias: 若为true则添加bias.
        input_is_parallel:  若为true，则认为输入应用被划分至各个GPU上，不需要进一步的划分.
        init_method: 随机初始化方法.
        stride: strided线性层.
    """

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        # 获得张量并行组的world_size
        world_size = get_tensor_model_parallel_world_size()

        # 按照张量并行度(world_size)划分输出维度
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result 执行 XA^T+b
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            # 初始化张量. 若完整权重矩阵A为n*m，张量并行度为k，这里初始化的张量为n*(m/k)
            # 也就是张量并行组中的进程各自初始化持有的部分张量
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=args.params_dtype))
            # 使用init_method对权重矩阵self.weight进行随机初始化(CPU版)
            # self.master_weight在测试中使用，这里不需要关注
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            # 使用init_method对权重矩阵self.weight进行随机初始化(GPU版)
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=1, stride=stride)
        if bias:
            # 实例化一个bias
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        self.bias_tp_auto_sync = args.sync_tp_duplicated_parameters

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            # 前向传播时，将input_分片至张量并行组中的各个进程中
            # 反向传播时，将张量并行组中持有的部分input_梯度合并为完整的梯度
            # 此时，_input是完整的输入张量，input_parallel则是分片后的张量，即input_parallel!=_input
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        # 对张量并行组中的输出进行allreduce，即操作X1A1+X2A2
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)

        if self.bias_tp_auto_sync:
            torch.distributed.all_reduce(self.bias, op=torch.distributed.ReduceOp.AVG,
                                         group=mpu.get_tensor_model_parallel_group())

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias


# layers.py
class MyVocabParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(MyVocabParallelEmbedding, self).__init__()
        # 初始化一些参数
        self.num_embeddings = num_embeddings # 词表大小
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # 张量并行组中的每个rank仅持有部分vocab embedding
        # 这里会计算当前rank持有的vocab的起始和结束位置
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        # 当前rank持有的部分vocab的大小
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        args = get_args()

        # embedding层添加LayerNorm
        if mpu.is_pipeline_first_stage() and (args.use_bnb_optimizer or args.embed_layernorm):
            self.norm = LayerNorm(embedding_dim)

        # bnb是指bitsandbytes，该库针对8-bit做了一些cuda函数的封装，这里忽略
        if args.use_bnb_optimizer:
            # for BNB we ignore the passed init_method and use torch.nn.init.xavier_uniform_
            # modified to calculate std on the unpartitioned embedding
            init_method = partial(xavier_uniform_tensor_parallel_, tp_degree=self.tensor_model_parallel_size)

        # 初始化embedding层的权重
        # 每个rank仅初始化自己所持有的那部分
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)
        # bnb(忽略)
        if args.use_bnb_optimizer:
            from bitsandbytes.optim import GlobalOptimManager
            GlobalOptimManager.get_instance().override_config(self.weight, 'optim_bits', 32)
            GlobalOptimManager.get_instance().register_parameters(self.weight)

    def forward(self, input_):
        if torch.any(input_ >= self.num_embeddings):
            raise ValueError(f"There is an input id in the input that is greater than the highest possible input id.\nInput: {input_}\nnum_embeddings: {self.num_embeddings}")
        # 全局rank
        global_rank = torch.distributed.get_rank()
        # 张量并行组中的rank
        tp_rank = get_tensor_model_parallel_rank()
        info = f"*"*20 + \
                f"\n> global_rank={global_rank}\n" + \
                f"> tensor parallel rank={tp_rank}\n" + \
                f"> full embedding size={(self.num_embeddings, self.embedding_dim)}\n" + \
                f"> partial embedding size={list(self.weight.size())}\n" \
                f"> input = {input_}\n" \
                f"> vocab_start_index={self.vocab_start_index}, vocab_end_index={self.vocab_end_index}\n"
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            # input_ is garanted to be in the range [0:self.vocab_end_index - self.vocab_start_index] thanks to the first check
            masked_input = input_
        info += f"> input_mask={input_mask} \n"
        info += f"> masked_input={masked_input} \n"

        # 获得embedding
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # 由于在当前rank上，仅能获得部分输入的embedding
        # 因此，将mask掉的input对应的embedding设置为全0
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        info += f"> output_parallel={output_parallel}\n"
        # 上一步设置为全0的embedding会在这一步通过allreduce，组装成完整的embedding
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        info += f"> output={output}\n"

        if hasattr(self, 'norm'):
            output = self.norm(output)
        print(info, end="")
        return output
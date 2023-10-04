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

import torch

from .initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from .utils import split_tensor_along_last_dim

# 1. _reduce
# 源代码
# _reduce提供在整个张量并行组进行All-Reduce的功能，函数定义如下 ：
def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    """
    在模型并行组上对输入张量执行All-reduce.
    """
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size()==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_

# 3. _split
# 源代码
# 沿最后一维分割张量，并保留对应rank的分片.
def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    """
    沿最后一维分割张量，并保留对应rank的分片.
    """
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # 按world_size分割输入张量input_
    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output

# 2. _gather
# 源代码
# 收集张量并行组中的张量，并按照最后一维度拼接.
def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""
    """
    gather张量并按照最后一维度拼接.
    """
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # 最后一维的索引
    # Size and dimension.
    last_dim = input_.dim() - 1

    # 张量并行组中的rank
    rank = get_tensor_model_parallel_rank()

    # 初始化空张量列表，用于存储收集来的张量
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # 拼接
    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output

# 前向传播时，不进行任何操作
# 反向传播时，对相同张量组中所有对input_的梯度求和
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):  # 前向传播时，不进行任何操作
        return input_

    @staticmethod
    def backward(ctx, grad_output):   # 反向传播时，对同张量并行组的梯度进行求和
        return _reduce(grad_output)

# 前向传播时，将同张量并行组的输入input_进行allreduce；
# 反向传播时，直接返回input_的梯度；
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):  # 前向传播时，对张量并行组中的输入进行allreduce
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# 前向传播时，将输入input_分片至同张量并行组的不同进程中；
# 反向传播时，将同张量并行组的梯度收集起来并拼接；
class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""
    """
    分割输入，仅保留对应rank的块。
    """
    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):  # 切分输入
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):  # 收集梯度
        return _gather(grad_output)

# 前向传播时，将同张量并行组的input_收集在一起并进行拼接；
# 反向传播时，将梯度分片至同张量并行组的不同进程中；
class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""
    """
    收集张量并行组的张量并拼接
    """
    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)
    
    @staticmethod
    def forward(ctx, input_):  # 前向传播时，相同张量并行组gather在一起
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):  # 反向传播时，将张量split至张量组中的机器
        return _split(grad_output)


# -----------------
# Helper functions.
# -----------------

def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)

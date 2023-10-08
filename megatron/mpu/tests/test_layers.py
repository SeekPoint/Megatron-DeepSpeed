# test_layers.py
import sys
sys.path.append("..")

import os
import torch.nn.functional as F
from megatron import get_args
from megatron.mpu import layers
from megatron.initialize import _initialize_distributed
from megatron.global_vars import set_global_variables
from commons import set_random_seed
from commons import print_separator
from commons import initialize_distributed
import megatron.mpu as mpu
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch
import random

class IdentityLayer2D(torch.nn.Module):
    """
    模拟一个输入为二维张量的神经网络
    """
    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight

def test_column_parallel_linear():
    global_rank = torch.distributed.get_rank()
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
    # 设置随机数种子
    seed = 12345
    set_random_seed(seed)
    # 张量并行组中，各个进程持有张量的input_size
    input_size_coeff = 4 #
    # 张量并行组中，各个进程持有张量的output_size
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 2
    output_size = output_size_coeff * tensor_model_parallel_size
    # 初始化一个产生二维张量的模拟网络，输入的张量为(batch_size, input_size)
    batch_size = 6
    identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    # 初始化一个列并行线性层
    linear_layer = mpu.ColumnParallelLinear(
        input_size, output_size, keep_master_weight_for_test=True, gather_output=False).cuda()
    # 随机初始化一个loss权重
    # 主要是为了计算标量的loss，从而验证梯度是否正确
    loss_weight = torch.randn([batch_size, output_size]).cuda()
    ## 前向传播
    input_ = identity_layer()
    # 此时，张量并行组中各个进程持有的output仅是完整输出张量的一部分
    output = linear_layer(input_)[0]

    if torch.distributed.get_rank() == 0:
        print(f"> Output size without tensor parallel is ({batch_size},{output_size})")
    torch.distributed.barrier()
    info = f"*"*20 + \
            f"\n> global_rank={global_rank}\n" + \
            f"> output size={output.size()}\n"
    print(info, end="")

class MyRowParallelLinear(mpu.RowParallelLinear):
    def forward(self, input_):
        global_rank = torch.distributed.get_rank()
        # 输入X，权重A和输出Y的形状
        X_size = list(input_.size())
        A_size = [self.input_size, self.output_size]
        Y_size = [X_size[0], A_size[1]]
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = mpu.scatter_to_tensor_model_parallel_region(input_)
        Xi_size = list(input_parallel.size())
        Ai_size = list(self.weight.T.size())

        info = f"*"*20 + \
                f"\n> global_rank={global_rank}\n" + \
                f"> size of X={X_size}\n" + \
                f"> size of A={A_size}\n" + \
                f"> size of Y={Y_size}\n" + \
                f"> size of Xi={Xi_size}\n" + \
                f"> size of Ai={Ai_size}\n"

        output_parallel = F.linear(input_parallel, self.weight)
        # 通过在output_parallel保证不同rank的output_parallel，便于观察后续的结果
        output_parallel = output_parallel + global_rank
        Yi_size = list(output_parallel.size())
        info += f"> size of Yi={Yi_size}\n" + \
                f"> Yi={output_parallel}\n"
        output_ = mpu.reduce_from_tensor_model_parallel_region(output_parallel)
        info += f"> Y={output_}"

        if self.bias_tp_auto_sync:
            torch.distributed.all_reduce(self.bias, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        print(info)
        return output, output_bias

def test_row_parallel_linear():
    global_rank = torch.distributed.get_rank()
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
    # 设置随机种子
    seed = 12345
    set_random_seed(seed)
     # 张量并行组中，各个进程持有张量的input_size
    input_size_coeff = 4
    input_size = input_size_coeff * tensor_model_parallel_size
     # 张量并行组中，各个进程持有张量的output_size
    output_size_coeff = 2
    output_size = output_size_coeff * tensor_model_parallel_size
    # 初始化一个产生二维张量的模拟网络，输入的张量为(batch_size, input_size)
    batch_size = 6
    identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    # 初始化一个行并行线性层
    linear_layer = MyRowParallelLinear(
        input_size, output_size, keep_master_weight_for_test=True).cuda()

    # 前向传播
    input_ = identity_layer()
    output = linear_layer(input_)

def main():
    set_global_variables(ignore_unknown_args=True)
    _initialize_distributed()
    world_size = torch.distributed.get_world_size()

    print_separator('Test test_column_parallel_linear')
    test_column_parallel_linear()

    print_separator('Test test_row_parallel_linear')
    test_row_parallel_linear()


if __name__ == '__main__':
    main()

'''
启动脚本

# 除了tensor-model-parallel-size和pipeline-model-parallel-size以外，
# 其余参数仅为了兼容原始代码，保存没有报错.
options=" \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 2 \
        --num-layers 10 \
        --hidden-size 768 \
        --micro-batch-size 2 \
        --num-attention-heads 32 \
        --seq-length 512 \
        --max-position-embeddings 512\
        --use_cpu_initialization True
        "

cmd="deepspeed test_layers.py $@ ${options}"

eval ${cmd}
'''
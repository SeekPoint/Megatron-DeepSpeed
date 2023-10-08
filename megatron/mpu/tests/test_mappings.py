
# test_mappings.py
import sys
sys.path.append("..")

from torch.nn.parameter import Parameter
from commons import print_separator
from commons import initialize_distributed
import megatron.mpu.mappings as mappings
import megatron.mpu as mpu
import torch

def test_reduce():
    print_separator(f'> Test _reduce')
    global_rank = torch.distributed.get_rank()
    # global_rank为1时，则会生成张量tensor([1])
    tensor = torch.Tensor([global_rank]).to(torch.device("cuda", global_rank))
    print(f"> Before reduce: {tensor}")
    # 保证reduce前后的输出不混乱
    torch.distributed.barrier()
    # reduce操作
    # 期望结果：[Rank0, Rank1]为一组，经过reduce后均为tensor([1])
    # 期望结果：[Rank6, Rank7]为一组，经过reduce后均为tensor([13])
    mappings._reduce(tensor)
    print(f"> After reduce: {tensor}")

def test_gather():
    print_separator(f'> Test _gather')
    global_rank = torch.distributed.get_rank()
    # global_rank为1时，则会生成张量tensor([1])
    tensor = torch.Tensor([global_rank]).to(torch.device("cuda", global_rank))
    print(f"> Before gather: {tensor}\n", end="")
    torch.distributed.barrier()
    # 期望结果：[Rank0, Rank1]为一组，经过gather后均为tensor([0., 1.])
    gather_tensor = mappings._gather(tensor)
    print(f"> After gather: {gather_tensor}\n", end="")

def test_split():
    print_separator(f'> Test _split')
    global_rank = torch.distributed.get_rank()

    # 在实验设置下为tp_world_size=2
    tp_world_size = mpu.get_tensor_model_parallel_world_size()

    # 在实验设置下tensor=[0,1]
    tensor = torch.Tensor(list(range(tp_world_size))).to(torch.device("cuda", global_rank))
    print(f"> Before split: {tensor}\n", end="")
    torch.distributed.barrier()

    # 期望结果：Rank0,Rank2,Rank4,Rank6持有张量tensor([0])
    # 期望结果：Rank1,Rank3,Rank5,Rank7持有张量tensor([1])
    split_tensor = mappings._split(tensor)
    print(f"> After split: {split_tensor}\n", end="")

def test_copy_to_tensor_model_parallel_region():
    print_separator(f'> Test copy_to_tensor_model_region')
    global_rank = torch.distributed.get_rank()

    # global_rank为1时，则会生成张量tensor([1])
    tensor = Parameter(torch.Tensor([global_rank]).to(torch.device("cuda", global_rank)))
    loss = global_rank * tensor
    loss.backward()

    # 非copy的tensor梯度期望结果为，Ranki的梯度为i
    print(f"> No copy grad: {tensor.grad}\n", end="")
    torch.distributed.barrier()
    tensor.grad = None

    # 使用copy_to_tensor_model_parallel_region对tensor进行操作
    # 该操作不会影响前向传播，仅影响反向传播
    tensor_parallel = mappings.copy_to_tensor_model_parallel_region(tensor)

    # 例：对于rank=5，则loss=5*x，其反向传播的梯度为5；依次类推
    loss_parallel = global_rank * tensor_parallel
    loss_parallel.backward()
    torch.distributed.barrier()

    # 例：张量组[Rank6, Rank7]的期望梯度均为13
    print(f"> Copy grad: {tensor.grad}\n", end="")

def test_reduce_from_tensor_model_parallel_region():
    print_separator(f"> Test reduce_from_tensor_model_parallel_region")
    global_rank = torch.distributed.get_rank()
    # global_rank为1时，则会生成张量tensor([1])
    tensor1 = Parameter(torch.Tensor([global_rank]).to(torch.device("cuda", global_rank)))
    tensor2 = global_rank * tensor1
    tensor_parallel = mappings.reduce_from_tensor_model_parallel_region(tensor2)
    loss = 2 * tensor_parallel
    loss.backward()
    print(f"> loss: {loss}\n", end="")
    print(f"> grad: {tensor1.grad}\n", end="")

def test_scatter_to_tensor_model_parallel_region():
    print_separator(f'> Test scatter_to_tensor_model_parallel_region')
    global_rank = torch.distributed.get_rank()
    tp_world_size = mpu.get_tensor_model_parallel_world_size()
    # tensor = [1,2]
    tensor = Parameter(torch.Tensor(list(range(1, tp_world_size+1))).to(torch.device("cuda", global_rank)))
    # split之后, Rank0、Rank2、Rank4、Rank6为tensor([1]), 其余Rank为tensor([2])
    tensor_split = mappings.scatter_to_tensor_model_parallel_region(tensor)
    loss = global_rank * tensor_split
    loss.backward()
    print(f"> Before split: {tensor}\n", end="")
    torch.distributed.barrier()
    print(f"> After split: {tensor_split}\n", end="")
    torch.distributed.barrier()
    print(f"> Grad: {tensor.grad}\n", end="")

def test_gather_from_tensor_model_parallel_region():
    print_separator(f'> Test gather_from_tensor_model_parallel_region')
    global_rank = torch.distributed.get_rank()
    # tp_world_size = mpu.get_tensor_model_parallel_world_size()
    tensor = Parameter(torch.Tensor([global_rank]).to(torch.device("cuda", global_rank)))
    print(f"> Before gather: {tensor}\n", end="")
    torch.distributed.barrier()
    # 例: [Rank6, Rank7]的gather_tensor均为tensor([6.,7.])
    gather_tensor = mappings.gather_from_tensor_model_parallel_region(tensor)
    print(f"> After gather: {gather_tensor.data}\n", end="")
    loss = (global_rank * gather_tensor).sum()
    loss.backward()
    print(f"> Grad: {tensor.grad}\n", end="")

if __name__ == '__main__':
    initialize_distributed()
    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = 2
    pipeline_model_parallel_size = 2
    # 并行环境初始化
    mpu.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size)
    test_reduce()
    test_gather()
    test_split()
    test_copy_to_tensor_model_parallel_region()
    test_reduce_from_tensor_model_parallel_region()
    test_scatter_to_tensor_model_parallel_region()
    test_gather_from_tensor_model_parallel_region()
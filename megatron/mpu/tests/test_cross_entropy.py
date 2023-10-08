# test_cross_entropy.py
import sys
sys.path.append("..")

from commons import set_random_seed
from commons import IdentityLayer
from commons import print_separator
from commons import initialize_distributed
from megatron.mpu.cross_entropy import _MyVocabParallelCrossEntropy
import megatron.mpu as mpu
import torch.nn.functional as F
import torch
import random

def test_cross_entropy():
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    batch_size = 32
    seq_length = 128
    vocab_size_per_partition = 500
    logits_scale = 1000.0
    vocab_size = vocab_size_per_partition * tensor_model_parallel_size
    seed = 1234

    set_random_seed(seed)
    identity = IdentityLayer((batch_size, seq_length, vocab_size),
                             scale=logits_scale).cuda()
    logits = identity()
    logits_parallel = mpu.scatter_to_tensor_model_parallel_region(logits)
    target = torch.cuda.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size)
    loss = _MyVocabParallelCrossEntropy.apply(logits_parallel, target).mean()

if __name__ == '__main__':
    initialize_distributed()
    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = 2
    pipeline_model_parallel_size = 2

    mpu.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size)

    test_cross_entropy()
# 启动命名：
#
# deepspeed test_cross_entropy.py
# 3. 测试结果
# ...

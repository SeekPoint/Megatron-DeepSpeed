# test_embedding.py
import sys
sys.path.append("..")

from megatron.mpu import layers
from commons import set_random_seed
from commons import print_separator
from megatron.initialize import _initialize_distributed
from megatron.global_vars import set_global_variables
import megatron.mpu as mpu
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch
import random

def test_parallel_embedding():
    batch_size = 2
    seq_length = 4
    vocab_size = 6
    hidden_size = 8
    seed = 123

    set_random_seed(seed)
    # (2,4)
    input_data = torch.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size).cuda()

    embedding_vocab_parallel = layers.MyVocabParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_).cuda()
    output = embedding_vocab_parallel(input_data)

def main():
    set_global_variables(ignore_unknown_args=True)
    _initialize_distributed()
    world_size = torch.distributed.get_world_size()

    print_separator('Test test_parallel_embedding')
    test_parallel_embedding()


if __name__ == '__main__':
    main()

'''
    启动命令：
    
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
    
    cmd="deepspeed test_embedding.py $@ ${options}"
    
    eval ${cmd}
3. 测试结果

全局rank为2，在张量并行组中的rank为0；
完整的embedding层大小应为(6, 8)，当前设备持有的embedding层大小为(3, 8)，符合张量并行度为2的假设；
当前设备持有的词表id范围介于0到3，输入中超出该词表范围都会被mask；
当前设备的输出(output_parallel)，会有部分embedding为全0，而完整的输出(output)则将张量并行组中所有的embedding输出都聚合在一起；
三、张量并行版交叉熵

'''
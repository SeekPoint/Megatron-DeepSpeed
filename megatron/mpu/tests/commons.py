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
# commons.py
import sys
sys.path.append("..")
import argparse
import os
import random
import numpy
import torch

from pydebug import debuginfo

import megatron.mpu as mpu # 由于本文测试代码位于项目的根目录下，因此修改了mpu的import方式

class IdentityLayer(torch.nn.Module):
    """
    一个单层网络，会在测试cross_entropy.py时使用。
    """
    def __init__(self, size, scale=1.0):
        super(IdentityLayer, self).__init__()
        self.weight = torch.nn.Parameter(scale * torch.randn(size))

    def forward(self):
        return self.weight


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    mpu.model_parallel_cuda_manual_seed(seed)


def initialize_distributed(backend='nccl'):
    """Initialize torch.distributed."""
    """初始化分布式环境"""
    # Get local rank in case it is provided.
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=None,
                        help='local rank passed from distributed launcher')
    args = parser.parse_args()
    local_rank = args.local_rank

    # Get rank and world size.
    rank = int(os.getenv('RANK', '0')) # 当前进程所对应的rank(rank是显卡所对应的全局编号)
    world_size = int(os.getenv("WORLD_SIZE", '1')) # world_size是指所有可用显卡的数量

    print('> initializing torch.distributed with local rank: {}, '
          'rank: {}, world size: {}'.format(local_rank, rank, world_size))

    # Set the device id.
    devices = torch.cuda.device_count()
    debuginfo(prj='mega_ds', info=f'torch.cuda.device_count() is: {devices}')

    device = rank % devices
    debuginfo(prj='mega_ds', info=f'device is: {device}')

    if local_rank is not None:
        device = local_rank
    torch.cuda.set_device(device)

    # Call the init process.
    # 初始化分布式环境所需要的相关代码
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', '192.168.1.11')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    debuginfo(prj='mega_ds', info=f'init_method is: {init_method}')
    torch.distributed.init_process_group(
        backend=backend, # 使用gpu时，backend最好选择nccl
        world_size=world_size,
        rank=rank,
        init_method=init_method)

def print_separator(message):
    """
    输出辅助函数
    """
    torch.distributed.barrier() # 保证所有进程在此处保存同步，主要是为了防止多进行输出混乱
    filler_len = (78 - len(message)) // 2
    filler = '-' * filler_len
    string = '\n' + filler + ' {} '.format(message) + filler
    if torch.distributed.get_rank() == 0:
        print(string, flush=True)
    torch.distributed.barrier()

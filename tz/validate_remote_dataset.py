import argparse
import os
import random
import shutil
import time
import warnings

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import grpc
from subprocess import Popen

from grpc_gen import example_meta_pb2, example_meta_pb2_grpc

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--track-cache-usage', action='store_true', help='to enable track cache usage')
# parser.add_argument('--server-host', default='127.0.0.1', type=str, help='server host')
best_acc1 = 0

server_host = "127.0.0.1"
port = 7890

uuid = os.environ["JOB_UUID"]
print("uuid: %s" % uuid)

server_address = "%s:%d" % (server_host, port)
channel = grpc.insecure_channel(server_address)
stub = example_meta_pb2_grpc.DatasetServiceStub(channel)
register_response = stub.Register(example_meta_pb2.RegisterRequest(uuid=uuid))
print(register_response)

example_paths = []
idx = 0

args = parser.parse_args()
reader_batch_size = int(args.batch_size / torch.cuda.device_count())

def path_reader(uuid):
    global example_paths
    global idx

    if example_paths is None or len(example_paths[idx:]) == 0:
        start_time = time.time()
        example_req = example_meta_pb2.ExampleRequest(num=reader_batch_size, worker_rank=0, uuid=uuid)
        example_paths = [exampleMeta.filepath.lstrip('/') for exampleMeta in stub.FetchExample(example_req)]; idx = 0
        print("rpc took %d" % (time.time() - start_time))

    ret = example_paths[idx]
    idx += 1
    return os.path.join('/data/', ret)

default_path_reader = path_reader


class ImageNetDataset(datasets.RemoteImageFolder):
    def __init__(self, root, transform=None, target_transform=None, uuid=None):
        super().__init__(root, transform, target_transform, uuid=uuid)
        # self.server_address = "%s:%d" % (server_host, port)
        # self.channel = grpc.insecure_channel(self.server_address)
        # self.stub = example_meta_pb2_grpc.DatasetServiceStub(self.channel)

    def get_path_reader(self):
        return default_path_reader


def main():
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    # traindir = os.path.join(args.data, 'data')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageNetDataset(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        uuid=uuid,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)




if __name__ == '__main__':
    main()
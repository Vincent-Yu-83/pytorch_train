import torch
from torch.cuda import max_memory_allocated
import torchvision
import argparse
import yaml
from torch.utils.data import DataLoader
from utils import ZeroOneNormalize, CosineAnnealingLRWarmup, evaluate_accuracy_and_loss
from matplotlib import pyplot as plt
import os
from transformers import get_cosine_schedule_with_warmup
import time
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler

'''
### **4卡 DDP（Distributed Data Parallel）**

[DISTRIBUTED COMMUNICATION PACKAGE - TORCH.DISTRIBUTED](https://pytorch.org/docs/stable/distributed.html)

[DISTRIBUTED DATA PARALLEL](https://pytorch.org/docs/stable/notes/ddp.html)

[[原创][深度][PyTorch] DDP系列第一篇：入门教程](https://zhuanlan.zhihu.com/p/178402798)

[[原创][深度][PyTorch] DDP系列第二篇：实现原理与源代码解析](https://zhuanlan.zhihu.com/p/187610959)

[[原创][深度][PyTorch] DDP系列第三篇：实战与技巧](https://zhuanlan.zhihu.com/p/250471767)

* 代码文件：pytorch_DDP.py  / pytorch_torchrun_DDP.py
* 单卡显存占用：3.12 G
* 单卡GPU使用率峰值：99%
* 训练时长（5 epoch）：560 s
* 训练结果：准确率85%左右
'''

os.environ["TORCH_HOME"] = "./pretrained_models"

# 创建命令行解析对象
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="./config/classifier_cifar10.yaml", type=str, help="data file path")
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()


def set_seed(seed):
    '''设置随机数种子'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# print(args.local_rank)
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)
# backend：指定分布式后端的名称，例如 ‘nccl’、‘gloo’ 或 ‘mpi’。
#   NCCL是NVIDIA集合通信库(NVIDIA Collective Communications Library)的简称,是用于加速多GPU之间通信的库,能够实现集合通信和点对点通信。
#   Open MPI项目是一个开源MPI（消息传递接口 ）实现，由学术，研究和行业合作伙伴联盟开发和维护。
#   Gloo是facebook开源的一套集体通信库，他提供了对机器学习中有用的一些集合通信算法如：barrier, broadcast, allreduce
# init_method：初始化方法的 URL 或文件路径。默认为 None，表示使用默认的初始化方法。
# timeout：初始化过程的超时时间，默认为 1800 秒。
# world_size：参与分布式训练的总进程数。默认为 -1，表示从环境变量中自动获取。
# rank：当前进程的排名。默认为 -1，表示从环境变量中自动获取。
# store：用于存储进程组信息的存储对象。默认为 None，表示使用默认存储。
# group_name：进程组的名称，默认为 ‘default’。
# **kwargs：其他可选参数，根据不同的分布式后端而定。
# torch.distributed.init_process_group(backend="nccl", rank=args.local_rank)
torch.distributed.init_process_group(backend="nccl")
world_size = torch.distributed.get_world_size()
set_seed(args.local_rank + 1)

cfg_path = args.cfg
with open(cfg_path, "r", encoding="utf8") as f:
    cfg_dict = yaml.safe_load(f)
print(cfg_dict)

# 显卡设备
visible_device = cfg_dict.get("device")
# 小批量
batchsize = cfg_dict.get("batch_size")
# worker数量
num_workers = cfg_dict.get("num_workers")
# epoch
num_epoches = cfg_dict.get("epoch")
# 学习率
lr = cfg_dict.get("lr")
# 权重衰减
weight_decay = cfg_dict.get("weight_decay")
# 存储目录
save_dir = cfg_dict.get("save_dir")

train_transforms_list = [
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize(size=(256, 256), antialias=True).cuda(),
    torchvision.transforms.RandomCrop(size=(224, 224)),
    ZeroOneNormalize(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
val_transforms_list = [
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize(size=(224, 224), antialias=True).cuda(),
    ZeroOneNormalize(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

train_transforms = torchvision.transforms.Compose(train_transforms_list)
val_transforms = torchvision.transforms.Compose(val_transforms_list)

if args.local_rank not in [-1, 0]:
    # 分布式任务汇聚点
    torch.distributed.barrier()

cifar10_train = torchvision.datasets.CIFAR10(root="./data", train=True, transform=train_transforms, download=True)
cifar10_test = torchvision.datasets.CIFAR10(root="./data", train=False, transform=val_transforms, download=True)

if args.local_rank == 0:
    # 分布式任务汇聚点
    torch.distributed.barrier()

cifar10_train_sampler = DistributedSampler(cifar10_train, shuffle=True, rank=args.local_rank, num_replicas=world_size)
cifar10_test_sampler = DistributedSampler(cifar10_test, shuffle=False, rank=args.local_rank, num_replicas=world_size)

train_data_loader = DataLoader(cifar10_train, batch_size=batchsize // len(visible_device), drop_last=True,
                               shuffle=False,
                               num_workers=num_workers,
                               sampler=cifar10_train_sampler)
test_data_loader = DataLoader(cifar10_test, batch_size=batchsize // len(visible_device), drop_last=False,
                              shuffle=False,
                              num_workers=num_workers,
                              sampler=cifar10_test_sampler)
classes = cifar10_train.classes
print("train: {}, test: {}, classes: {}".format(len(train_data_loader), len(test_data_loader), len(classes)))

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
loss = torch.nn.CrossEntropyLoss()
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10,
                                               num_training_steps=len(train_data_loader) * num_epoches)

if __name__ == '__main__':
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    lr_decay_list = []
    memory = 0
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    best_acc = 0.0
    best_model = ""
    start_time = time.time()
    for epoch in range(num_epoches):
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        n = 0
        model.train()

        # set_epoch是为了让不同的epoch采样出不同的样本顺序，同时，保证同一个人epoch下，各个进程之间的相同随机数种子
        # 这样就可以根据rank编号和world_size使不同进程获取到不重复的数据
        cifar10_train_sampler.set_epoch(epoch)

        for batch_idx, (X, y) in enumerate(train_data_loader):
            lr_decay_list.append(optimizer.state_dict()["param_groups"][0]["lr"])
            # print(lr_decay_list)

            X = X.cuda()
            y = y.cuda()
            y_pred = model(X)
            l = loss(y_pred, y).sum()
            # print("local rank: {}, {}, {}, {}".format(args.local_rank, X.shape, y.shape, y_pred.shape))

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss_sum += l.item()
            train_acc_sum += (y_pred.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

            batch_acc = (y_pred.argmax(dim=1) == y).float().mean()
            # 跨分布式汇总，同步梯度
            torch.distributed.all_reduce(batch_acc, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(l, op=torch.distributed.ReduceOp.AVG)

            # 收集所有数据块并分发到所有rank上。不计算梯度
            # X_gather = torch.zeros_like(X).repeat((world_size, 1, 1, 1))
            # y_gather = torch.zeros_like(y).repeat(world_size)
            # y_pred_gather = torch.zeros_like(y_pred).repeat((world_size, 1))
            # torch.distributed.all_gather_into_tensor(X_gather, X)
            # torch.distributed.all_gather_into_tensor(y_gather, y)
            # torch.distributed.all_gather_into_tensor(y_pred_gather, y_pred)
            # print("X_gather: {}, y_gather: {}, y_pred_gather: {}".format(X_gather.shape, y_gather.shape,
            #                                                              y_pred_gather.shape))

            if batch_idx % 20 == 0 and args.local_rank == 0:
                print("epoch: {}, iter: {}, iter loss: {:.4f}, iter acc: {:.4f}".format(epoch, batch_idx, l.item(),
                                                                                        batch_acc.item()))
            lr_scheduler.step()

        model.eval()
        v_acc, v_loss = evaluate_accuracy_and_loss(test_data_loader, model, loss, accelerator=None, is_half=False,
                                                local_rank=args.local_rank,
                                                world_size=world_size)
        train_acc.append(train_acc_sum / n)
        train_loss.append(train_loss_sum / n)
        val_acc.append(v_acc)
        val_loss.append(v_loss)

        # 分布式任务汇聚点
        torch.distributed.barrier()
        if args.local_rank == 0:
            if v_acc > best_acc:
                if os.path.exists(os.path.join(save_dir, file_name)) is False:
                    os.makedirs(os.path.join(save_dir, file_name))
                best_acc = v_acc
                best_model = os.path.join(os.path.join(save_dir, file_name),
                                        "{}-{}-{}.pth".format(file_name, epoch, best_acc))
                torch.save(model.module.state_dict(), best_model)

            print("epoch: {}, train acc: {:.4f}, train loss: {:.4f}, val acc: {:.4f}, val loss: {:.4f}".format(
                epoch, train_acc[-1], train_loss[-1], val_acc[-1], val_loss[-1]))
        
        # PyTorch 提供了 memory_allocated() 和 max_memory_allocated() 用于监视 tensors 占用的内存； 
        # memory_cached() 和 max_memory_cached() 用于监视缓存分配器所管理的内存.
        memory = max_memory_allocated()
        print(f'memory allocated: {memory / 1e9:.2f}G')
    end_time = time.time()
    duration = int(end_time - start_time)
    print("duration time: {} s".format(duration))

    if args.local_rank == 0:
        fig, axes = plt.subplots(1, 3)
        axes[0].plot(list(range(1, num_epoches + 1)), train_loss, color="r", label="train loss")
        axes[0].plot(list(range(1, num_epoches + 1)), val_loss, color="b", label="validate loss")
        axes[0].legend()
        axes[0].set_title("Loss")

        axes[1].plot(list(range(1, num_epoches + 1)), train_acc, color="r", label="train acc")
        axes[1].plot(list(range(1, num_epoches + 1)), val_acc, color="b", label="validate acc")
        axes[1].legend()
        axes[1].set_title("Accuracy")

        axes[2].plot(list(range(1, len(lr_decay_list) + 1)), lr_decay_list, color="r", label="lr")
        axes[2].legend()
        axes[2].set_title("Learning Rate")
        plt.suptitle('memory: {:.2f} G , duration: {} s'.format(memory / 1e9, duration))
        plt.savefig(os.path.join(save_dir, "{}.jpg".format(file_name)))
        plt.show()

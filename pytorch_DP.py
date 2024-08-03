import time
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

'''
### **4卡 DP（Data Parallel）**
* 代码文件：pytorch_DP.py
* 单卡显存占用：3.08 G
* 单卡GPU使用率峰值：99%
* 训练时长（5 epoch）：942 s
* 训练结果：准确率85%左右
'''

# 定义torch目录
os.environ["TORCH_HOME"] = "./pretrained_models"

# 读取cifar10数据集配置
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="./config/classifier_cifar10.yaml", type=str, help="data file path")
args = parser.parse_args()
cfg_path = args.cfg

# load配置
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

# 训练集
train_transforms_list = [
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize(size=(256, 256), antialias=True).cuda(),
    torchvision.transforms.RandomCrop(size=(224, 224)),
    ZeroOneNormalize(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
# 验证集
val_transforms_list = [
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize(size=(224, 224), antialias=True).cuda(),
    ZeroOneNormalize(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

train_transforms = torchvision.transforms.Compose(train_transforms_list)
val_transforms = torchvision.transforms.Compose(val_transforms_list)

# 下载CIFAR10训练数据集
cifar10_train = torchvision.datasets.CIFAR10(root="./data", train=True, transform=train_transforms, download=True)
# 下载CIFAR10测试数据集
cifar10_test = torchvision.datasets.CIFAR10(root="./data", train=False, transform=val_transforms, download=True)

train_data_loader = DataLoader(cifar10_train, batch_size=batchsize, drop_last=True, shuffle=True,
                               num_workers=num_workers)
test_data_loader = DataLoader(cifar10_test, batch_size=batchsize, drop_last=False, shuffle=False,
                              num_workers=num_workers)
classes = cifar10_train.classes
print("train: {}, test: {}, classes: {}".format(len(train_data_loader), len(test_data_loader), len(classes)))

# 初始化网络（resnet50）使用IMAGENET1K_V1初始化权重，存入GPU
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda()
model = torch.nn.DataParallel(model, device_ids=visible_device)
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
    # 开始训练
    for epoch in range(num_epoches):
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        n = 0
        model.train()

        for batch_idx, (X, y) in enumerate(train_data_loader):
            lr_decay_list.append(optimizer.state_dict()["param_groups"][0]["lr"])
            # print(lr_decay_list)
            # 在GPU上完成损失计算
            X = X.cuda()
            y = y.cuda()
            y_pred = model(X)
            l = loss(y_pred, y).sum()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # 训练损失累计
            train_loss_sum += l.item()
            # 训练成功累计
            train_acc_sum += (y_pred.argmax(dim=1) == y).sum().item()
            # 真实值累加
            n += y.shape[0]

            if batch_idx % 20 == 0:
                print("epoch: {}, iter: {}, iter loss: {:.4f}, iter acc: {:.4f}".format(epoch, batch_idx, l.item(), (
                        y_pred.argmax(dim=1) == y).float().mean().item()))
            # 更新学习率
            lr_scheduler.step()
        
        # 评估模式，验证模型
        model.eval()
        v_acc, v_loss = evaluate_accuracy_and_loss(test_data_loader, model, loss, accelerator=None)
        train_acc.append(train_acc_sum / n)
        train_loss.append(train_loss_sum / n)
        val_acc.append(v_acc)
        val_loss.append(v_loss)

        # 如果验证结果比最好结果还好，则更新最好结果，保存模型
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

    # 存储运行结果图
    plt.suptitle('memory: {:.2f} G , duration: {} s'.format(memory / 1e9, duration))
    plt.savefig(os.path.join(save_dir, "{}.jpg".format(file_name)))
    plt.show()

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

'''
### **单卡混合精度训练**

[AUTOMATIC MIXED PRECISION PACKAGE - TORCH.AMP](https://pytorch.org/docs/stable/amp.html#torch.autocast)

[CUDA AUTOMATIC MIXED PRECISION EXAMPLES](https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples)

[PyTorch 源码解读之 torch.cuda.amp: 自动混合精度详解](https://zhuanlan.zhihu.com/p/348554267)

[如何使用 PyTorch 进行半精度、混(合)精度训练](https://blog.csdn.net/qq_44089890/article/details/130471991)

[如何使用 PyTorch 进行半精度训练](https://blog.csdn.net/qq_39845931/article/details/121671342)

[pytorch模型训练之fp16、apm、多GPU模型、梯度检查点（gradient checkpointing）显存优化等](https://zhuanlan.zhihu.com/p/448395808)

[Working with Multiple GPUs](https://pytorch.org/docs/stable/notes/amp_examples.html#amp-multigpu)

* 代码文件：pytorch_auto_mixed_precision.py
* 单卡显存占用：6.02 G
* 单卡GPU使用率峰值：100%
* 训练时长（5 epoch）：1546 s
* 训练结果：准确率85%左右
'''

os.environ["TORCH_HOME"] = "./pretrained_models"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 创建命令行解析对象
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="./config/classifier_cifar10.yaml", type=str, help="data file path")
args = parser.parse_args()
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
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
loss = torch.nn.CrossEntropyLoss()
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10,
                                               num_training_steps=len(train_data_loader) * num_epoches)

if __name__ == '__main__':
    # 是PyTorch中一种自动混合精度计算的方法，它允许在深度学习模型的训练过程中自动执行混合精度计算，从而加快训练速度并减少显存占用。

    # 在使用torch.cuda.amp.autocast时，一般会将模型的前向传播和反向传播包裹在with torch.cuda.amp.autocast()上下文中，
    # 以指示PyTorch使用混合精度计算。在这个上下文中，PyTorch会自动将部分计算转换为半精度浮点数（FP16），以提高计算速度和减少显存使用。
    autocast = torch.cuda.amp.autocast
    # Nvidia 在 Volta 架构中引入 Tensor Core 单元，来支持 FP32 和 FP16 混合精度计算。
    # 也在 2018 年提出一个 PyTorch 拓展 apex，来支持模型参数自动混合精度训练。
    # 自动混合精度（Automatic Mixed Precision, AMP)训练，是在训练一个数值精度 FP32 的模型，一部分算子的操作时，数值精度为 FP16，
    # 其余算子的操作精度是 FP32，而具体哪些算子用 FP16，哪些用 FP32，不需要用户关心，amp 自动给它们都安排好了。
    # 这样在不改变模型、不降低模型训练精度的前提下，可以缩短训练时间，降低存储需求，因而能支持更多的 batch size、更大模型和尺寸更大的输入进行训练。
    # PyTorch 从 1.6 以后（在此之前 OpenMMLab 已经支持混合精度训练，即 Fp16OptimizerHook），开始原生支持 amp，即torch.cuda.amp module。
    # 2020 ECCV，英伟达官方做了一个 tutorial 推广 amp。从官方各种文档网页 claim 的结果来看，
    # amp 在分类、检测、图像生成、3D CNNs、LSTM，以及 NLP 中机器翻译、语义识别等应用中，都在没有降低模型训练精度都前提下，加速了模型的训练速度。
    # 用户不需要手动对模型参数 dtype 转换，amp 会自动为算子选择合适的数值精度
    # 对于反向传播的时候，FP16 的梯度数值溢出的问题，amp 提供了梯度 scaling 操作，而且在优化器更新参数前，会自动对梯度 unscaling，所以，对用于模型优化的超参数不会有任何影响
    # 以上两点，分别是通过使用amp.autocast和amp.GradScaler来实现的。
    scaler = torch.cuda.amp.GradScaler()
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

        for batch_idx, (X, y) in enumerate(train_data_loader):
            lr_decay_list.append(optimizer.state_dict()["param_groups"][0]["lr"])
            # print(lr_decay_list)

            X = X.cuda()
            y = y.cuda()

            with autocast():
                y_pred = model(X)
                l = loss(y_pred, y).sum()

            optimizer.zero_grad()
            
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            l = scaler.scale(l)
            l.backward()
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            # 训练损失累计
            train_loss_sum += l.item()
            # 训练成功累计
            train_acc_sum += (y_pred.argmax(dim=1) == y).sum().item()
            # 真实值累加
            n += y.shape[0]

            # if batch_idx > 100:
            #     break

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
            torch.save(model.state_dict(), best_model)

        print("epoch: {}, train acc: {:.4f}, train loss: {:.4f}, val acc: {:.4f}, val loss: {:.4f}".format(
            epoch, train_acc[-1], train_loss[-1], val_acc[-1], val_loss[-1]))
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
    plt.suptitle('memory: {:.2f} G , duration: {} s'.format(memory / 1e9, duration))
    plt.savefig(os.path.join(save_dir, "{}.jpg".format(file_name)))
    plt.show()

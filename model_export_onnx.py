# coding:utf-8
import os

import torch
import onnx
import onnxruntime
import numpy as np
import time
import torchvision

os.environ["TORCH_HOME"] = "./pretrained_models"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def pytorch_2_onnx(model_path, export_model_path):
    """
    将pytorch模型导出为onnx，导出时pytorch内部使用的是trace或者script先执行一次模型推理，然后记录下网络图结构
    所以，要导出的模型要能够被trace或者script进行转换
    :return:
    """
    # 加载预训练模型
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda()
    # print(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    # pytorch转换为onnx内部使用trace或者script，需要提供一组输入数据执行一次模型推理过程，然后进行trace记录
    dummy_input = torch.randn(4, 3, 224, 224, device="cuda")
    input_names = ["input_data"] + ["learned_%d" % i for i in range(16)]
    output_names = ["output_data"]

    torch.onnx.export(
        model,  # pytorch网络模型
        dummy_input,  # 随机的模拟输入
        export_model_path,  # 导出的onnx文件位置
        export_params=True,  # 导出训练好的模型参数
        verbose=10,  # debug message
        training=torch.onnx.TrainingMode.EVAL,  # 导出模型调整到推理状态，将dropout，BatchNorm等涉及的超参数固定
        input_names=input_names,  # 为静态网络图中的输入节点设置别名，在进行onnx推理时，将input_names字段与输入数据绑定
        output_names=output_names,  # 为输出节点设置别名
        # 如果不设置dynamic_axes，那么对于输入形状为[4, 3, 224, 224]，在以后使用onnx进行推理时也必须输入[4, 3, 224, 224]
        # 下面设置了输入的第0维是动态的，以后推理时batch_size的大小可以是其他动态值
        dynamic_axes={
            # a dictionary to specify dynamic axes of input/output
            # each key must also be provided in input_names or output_names
            "input_data": {0: "batch_size"},
            "output_data": {0: "batch_size"}
        })
    return export_model_path


def onnx_check(model_path):
    """
    验证导出的模型格式时候正确
    :param model_path:
    :return:
    """
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))


def onnx_inference(model_path):
    """
    模型推理
    :param model_path:
    :return:
    """
    # 使用onnxruntime-gpu在GPU上进行推理
    session = onnxruntime.InferenceSession(model_path,
                                           providers=[
                                               ("CUDAExecutionProvider", {  # 使用GPU推理
                                                   "device_id": 0,
                                                   "arena_extend_strategy": "kNextPowerOfTwo",
                                                   "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                                                   "cudnn_conv_algo_search": "EXHAUSTIVE",
                                                   "do_copy_in_default_stream": True,
                                                   # "cudnn_conv_use_max_workspace": "1"    # 在初始化阶段需要占用好几G的显存
                                               }),
                                               "CPUExecutionProvider"  # 使用CPU推理
                                           ])
    # session = onnxruntime.InferenceSession(model_path)
    data = np.random.randn(2, 3, 224, 224).astype(np.float32)
    X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(data,'cuda',0)
    print("ortvalue device:",X_ortvalue.device_name())  # 获取ortvalue对象所在的设备名称
    print("ortvalue data type:",X_ortvalue.data_type())  # 获取ortvalue对象的数据类型
    print("ortvalue shape:",X_ortvalue.shape())  # 获取ortvalue对象的形状
    print("if ortvalue is tensor:",X_ortvalue.is_tensor())  # 判断ortvalue对象是否是tensor对象
    print(np.array_equal(X_ortvalue.numpy(), data))  # 判断ortvalue对象的数据是否与numpy数组相等


    # 获取模型原始输入的字段名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("input name: {}".format(input_name))

    # 以字典方式将数据输入到模型中
    outputs = session.run([output_name], {input_name: data})
    print('outputs: ', outputs)
    
    '''
    绑定输入输出，通过io_binding来进行推理。
    绑定时需要设置好输入输出的shape，否则会报错，输出的绑定可以不用管，onnxruntime会自动进行推测输出的shape并在设备上分配内存给输出。

    个人理解这种io_binding做法的好处就是固定好输入输出的内存地址，避免不断allocated和free的过程，更加高效。有点像设计模式中的单例模式。
    '''
    io_binding = session.io_binding() # 创建io_binding对象
    io_binding.bind_input(input_name,
                        device_type=X_ortvalue.device_name(),
                        device_id=0,
                        element_type=np.float32,
                        shape=X_ortvalue.shape(),
                        buffer_ptr=X_ortvalue.data_ptr()) # 绑定模型的输入
    # buffer_ptr参数为tensor对象的数据指针，可以通过tensor对象的data_ptr方法获取
    io_binding.bind_output(output_name) # 绑定模型的输出
    #onnxruntime可以为output动态分配内存，也可以通过bind_output方法指定output的形状和数据类型
    session.run_with_iobinding(io_binding) # 使用io_binding方法进行模型推理

    Y = io_binding.copy_outputs_to_cpu()[0] # 将模型的输出从GPU上拷贝到CPU上
    print("Y shape:", Y.shape)
    
    #onnxruntime io_binding方法可以提高模型推理的效率，特别是在模型的输入和输出形状不变的情况下
    #onnxruntime io_binding还可以绑定到pytorch的tensor对象上，可以通过pytorch的tensor对象的data_ptr方法获取数据指针
    Y_tensor = torch.zeros(2,1000).cuda()
    io_binding.bind_output(
        output_name,
        device_type=Y_tensor.device.type,
        device_id=Y_tensor.device.index,
        element_type=np.float32,
        shape=Y_tensor.shape,
        buffer_ptr=Y_tensor.data_ptr())
    session.run_with_iobinding(io_binding)
    print("Y_tensor shape:", Y_tensor.shape)



if __name__ == '__main__':
    model_path = pytorch_2_onnx("results/pytorch_SingleGPU/pytorch_SingleGPU-4-0.8491.pth", "results/pytorch_SingleGPU/pytorch_SingleGPU.onnx")

    # onnx_check(model_path)

    onnx_inference(model_path)

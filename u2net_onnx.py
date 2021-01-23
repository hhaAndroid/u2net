# -*- coding: utf-8 -*-
# ======================================================
# @Time    : 21-1-23 下午7:10
# @Author  : huang ha
# @Email   : huang_ha@rr.com
# @File    : u2net_onnx.py
# @Comment: 
# ======================================================
import os

import torch
from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB


# 需要安装环境：  onnxruntime 安装的是cpu版本
# pip install onnx onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
# 如果想GPU运行，则 pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
# gpu版本需要有底层库支持


if __name__ == '__main__':
    model_name = 'u2net'  # u2net
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')


    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    net.load_state_dict(torch.load(model_dir))
    net.eval()

    x = torch.randn(1, 3, 320, 320, requires_grad=True)
    torch_out = net(x)

    torch.onnx.export(net,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      model_name+".onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})





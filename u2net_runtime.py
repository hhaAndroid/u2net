# -*- coding: utf-8 -*-
# ======================================================
# @Time    : 21-1-23 下午7:21
# @Author  : huang ha
# @Email   : huang_ha@rr.com
# @File    : u2net_runtime.py
# @Comment: 
# ======================================================

import glob
import os

import onnx
import onnxruntime
import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# normalize the predicted SOD probability map
def normPRED(d):
    ma = np.max(d)
    mi = np.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()

    im = Image.fromarray(predict * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    aaa = img_name.split(".")
    # bbb = aaa[0:-1]
    # imidx = bbb[0]
    # for i in range(1,len(bbb)):
    #     imidx = imidx + "." + bbb[i]

    imo.save(os.path.join(d_dir, aaa[0] + '_mask.' + aaa[1]))


if __name__ == '__main__':
    onnx_model = onnx.load("u2net.onnx")
    onnx.checker.check_model(onnx_model)

    image_dir = '/home/hha/dataset/circle/20210117'
    prediction_dir = '/home/hha/dataset/circle/20210117_1'

    img_name_list = glob.glob(image_dir + os.sep + '*')
    img_name_list = list(filter(lambda f: f.find('_mask') < 0, img_name_list))

    ort_session = onnxruntime.InferenceSession("u2net.onnx")

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),  # 320
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs_test)}
        ort_outs = ort_session.run(None, ort_inputs)

        pred = ort_outs[0][:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)








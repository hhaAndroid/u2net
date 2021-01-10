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
import cv2
import math

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir,thresh=0.5):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    pred_mask=np.where(predict_np>thresh,255,0).astype(np.uint8)

    pred_mask=cv2.cvtColor(pred_mask,cv2.COLOR_GRAY2BGR)
    img=cv2.imread(image_name)
    pred_mask=cv2.resize(pred_mask,img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)

    # pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
    # circles = cv2.HoughCircles(pred_mask, cv2.HOUGH_GRADIENT, 1.2, 100)
    # circles = circles[0, :, :]
    # circles = np.uint16(np.around(circles))
    # for i in circles[0, :]:
    #     # 画出外边圆
    #     cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # 画出圆心
    #     cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    # 轮廓查找
    ret, thresh = cv2.threshold(pred_mask[:,:,0], 127, 255, 0)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        if len(cnt)<5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        print(ellipse)

        cv2.ellipse(img, ellipse,(0, 255, 0), 2)
        # cv2.circle(img, (np.int(x), np.int(y)), int(radius), (255, 0, 0), 2, 8, 0)

        points = np.array(cnt).reshape((-1, 2)).astype(np.int32)
        cv2.polylines(img, [points], 2, (0,0,255))


    cv2.imshow('win',img)
    cv2.waitKey(0)


    # im = Image.fromarray(predict_np * 255).convert('RGB')
    # img_name = image_name.split(os.sep)[-1]
    # image = io.imread(image_name)
    # imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    #
    # aaa = img_name.split(".")
    #
    # imo.save(os.path.join(d_dir, aaa[0] + '_mask.' + aaa[1]))


def main():
    # --------- 1. get image path and name ---------
    model_name = 'u2netp'  # u2net

    # image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    image_dir = '/home/hha/dataset/circle/circle'
    prediction_dir = '/home/hha/dataset/circle/circle_pred'
    model_dir='/home/hha/pytorch_code/U-2-Net-master/saved_models/u2netp/u2netp.pthu2netp_bce_itr_2000_train_0.077763_tar_0.006976.pth'
    # model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')

    img_name_list = list(filter(lambda f: f.find('_mask') < 0, img_name_list))

    # print(img_name_list)

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
                                        num_workers=0)

    # --------- 3. model define ---------
    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = inputs_test.cuda()
        else:
            inputs_test = inputs_test

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        # pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7


if __name__ == "__main__":
    main()

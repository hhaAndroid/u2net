import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))

    return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2netp' #'u2net'

data_dir = '/home/hha/dataset/circle/circle'

img_list = glob.glob(data_dir + os.sep + '*')
tra_img_name_list=list(filter(lambda f: f.find('_mask') < 0, img_list))
tra_lbl_name_list=[]

for tra_img in tra_img_name_list:
    fname, name1 = os.path.split(tra_img)
    aaa = name1.split(".")
    mask_path = os.path.join(fname, aaa[0] + '_mask.' + aaa[1])
    tra_lbl_name_list.append(mask_path)


epoch_num = 100
batch_size_train = 12
batch_size_val = 1
train_num = 0
val_num = 0


print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2)

# ------- 3. define model --------
# define the net
if model_name== 'u2net':
    net = U2NET(3, 1)
elif model_name== 'u2netp':
    net = U2NETP(3,1)

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
net.load_state_dict(torch.load(model_dir))


def foze_encoder(net):
    net.stage1.eval()
    for param in net.stage1.parameters():
        param.requires_grad = False
    net.stage2.eval()
    for param in net.stage2.parameters():
        param.requires_grad = False
    net.stage3.eval()
    for param in net.stage3.parameters():
        param.requires_grad = False
    net.stage4.eval()
    for param in net.stage4.parameters():
        param.requires_grad = False
    net.stage5.eval()
    for param in net.stage5.parameters():
        param.requires_grad = False
    net.stage6.eval()
    for param in net.stage6.parameters():
        param.requires_grad = False


# 固定骨架
foze_encoder(net)


if torch.cuda.is_available():
    net.cuda()


# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=4)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000 # save the model every 2000 iterations

for epoch in range(0, epoch_num):
    net.train()
    foze_encoder(net)

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = inputs.cuda(),labels.cuda()

        else:
            inputs_v, labels_v = inputs, labels

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.item()
        running_tar_loss += loss2.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:

            torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0


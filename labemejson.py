# -*- coding: utf-8 -*-
# ======================================================
# @Time    : 21-1-9 上午11:10
# @Author  : huang ha
# @Email   : huang_ha@rr.com
# @File    : labemejson.py
# @Comment: 
# ======================================================
import glob
import os
import json
import cv2
import numpy as np


def ReadLabelMeJson(path):
    with open(path, "r", encoding='utf-8') as f:
        data=json.load(f)
    return data

def WriteJson(data,path):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data,f)


def visLabelMe(data,img):
    img=img.copy()
    shapes=data['shapes'] # n个物体
    for shape in shapes:
        points=shape['points']
        points=np.array(points).reshape((-1,2)).astype(np.int32)
        cv2.polylines(img, [points], 2, 255)


    cv2.namedWindow('win')
    cv2.imshow('win',img)
    cv2.waitKey(0)



def ConvertLabelMeToMask(data,img,img_path,is_save=False,is_show=False):
    mask=np.zeros(img.shape[:2],np.uint8)
    shapes = data['shapes']  # n个物体
    for shape in shapes:
        points = shape['points']
        points = np.array(points).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)

    if is_show:
        cv2.namedWindow('win')
        cv2.imshow('win', mask)
        cv2.waitKey(0)
    if is_save:
        fname,name=os.path.split(img_path)
        aaa = name.split(".")
        mask_path=os.path.join(fname,aaa[0]+'_mask.'+aaa[1])
        cv2.imwrite(mask_path,mask)


def ConvertMaskToLabelMe(mask_path,image_path):
    mask=cv2.imread(mask_path,0)

    imagePath=os.path.split(image_path)[-1]
    imageHeight =mask.shape[0]
    imageWidth =mask.shape[1]
    save_data = dict(imageData=None, version="4.5.6", imagePath=imagePath, flags={}, imageHeight=imageHeight,
                     imageWidth=imageWidth)

    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes=[]
    for cnt in cnts:
        points = dict(label="1", flags={}, group_id=None, shape_type="polygon")
        points1=np.array(cnt).reshape((-1,2)).astype(np.int32)
        points1=points1.tolist()
        points['points']=points1
        shapes.append(points)

    save_data['shapes']=shapes

    WriteJson(save_data,image_path[:-3]+'json')


if __name__ == '__main__':
    path='/home/hha/dataset/circle/labelme'
    img_list = glob.glob(path + os.sep + '*')
    img_name_list = list(filter(lambda f: f.find('.json') < 0 and f.find('_mask') < 0, img_list))
    print(img_name_list)
    for name in img_name_list:
        img = cv2.imread(name)

        json_path=name[:-3]+'json'

        fname, name1 = os.path.split(name)
        aaa = name1.split(".")
        mask_path = os.path.join(fname, aaa[0] + '_mask.' + aaa[1])

        # data=ReadLabelMeJson(json_path)

        # visLabelMe(data,img)

        # ConvertLabelMeToMask(data,img,name,is_save=True)

        ConvertMaskToLabelMe(mask_path,name)


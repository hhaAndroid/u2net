# -*- coding: utf-8 -*-
# ======================================================
# @Time    : 21-1-9 下午7:41
# @Author  : huang ha
# @Email   : huang_ha@rr.com
# @File    : rename_file.py
# @Comment: 
# ======================================================

import os


if __name__ == '__main__':
    image_dir = '/home/hha/dataset/circle/circle'
    files = os.listdir(image_dir)
    count=0
    for name in files:
        _, name1 = os.path.split(name)
        aaa = name1.split(".")
        NewName = os.path.join(image_dir, str(count)+'.'+aaa[1])
        OldName = os.path.join(image_dir, name)
        os.rename(OldName, NewName)
        count +=1
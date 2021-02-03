import os.path as osp

import cv2
import mmcv
import numpy as np


if __name__ == '__main__':
    out_dir = '/home/SENSETIME/huanghaian/dataset/project/unet'
    paths = mmcv.scandir(out_dir, '.json')
    for i, path in enumerate(paths):
        img_path = osp.join(out_dir, path[:-4] + 'jpg')
        image = mmcv.imread(img_path)
        json_path = osp.join(out_dir, path)
        json_data = mmcv.load(json_path)
        points = np.array(json_data['points']).reshape(-1, 2)
        print(points)

        for point in points:
            cv2.circle(image, tuple(point), 5, (255, 0, 0), -1)

        cv2.namedWindow('img', 0)
        mmcv.imshow(image, 'img')

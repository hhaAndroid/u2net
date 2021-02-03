import os.path as osp

import cv2
import mmcv
import numpy as np
import random


def _generate_batch_data(sampler, batch_size=5):
    batch = []
    for idx in sampler:
        batch.append(idx)
        if len(batch) == batch_size:
            yield batch
            batch = []


if __name__ == '__main__':
    root_path = '/home/SENSETIME/huanghaian/dataset/project/images'

    out_dir = '/home/SENSETIME/huanghaian/dataset/project/unet'
    mmcv.mkdir_or_exist(out_dir)

    paths = mmcv.scandir(root_path, '.json', recursive=True)
    count = 0
    for i, path in enumerate(paths):
        img_path = osp.join(root_path, path[:-4] + 'jpg')
        print(img_path)
        image = mmcv.imread(img_path)
        h, w = image.shape[:2]

        json_path = osp.join(root_path, path)
        json_data = mmcv.load(json_path)
        shapes = json_data['shapes']
        assert len(shapes) % 5 == 0

        batch_data = _generate_batch_data(shapes, 5)

        for data in batch_data:
            x1, y1 = data[0]['points'][0]
            x2, y2 = data[0]['points'][1]

            label = data[0]['label']
            if label == 'out-ng':
                continue

            tempoints = []
            tempoints.extend(data[1]['points'])
            tempoints.extend(data[2]['points'])
            tempoints.extend(data[3]['points'])
            tempoints.extend(data[4]['points'])
            points = np.array(tempoints, np.int32).reshape(4, 2)

            # crop
            for i in range(2):
                # expand
                expand_x = random.randint(10, 30)
                expand_y = random.randint(10, 30)
                y11 = max(int(y1) - expand_y, 0)
                y21 = min(int(y2) + expand_y, h)
                x11 = max(int(x1) - expand_x, 0)
                x21 = min(int(x2) + expand_x, w)
                img = image[y11:y21, x11:x21, :]
                points1 = points - np.array([x11, y11], np.int32).reshape(1, 2)

                save_img_path = osp.join(out_dir, str(count) + '.jpg')
                json_path = osp.join(out_dir, str(count) + '.json')
                points1 = points1.reshape(-1).tolist()
                json_data = {'points': points1}
                mmcv.imwrite(img, save_img_path)
                mmcv.dump(json_data, json_path)

                count += 1

                # for point in points1:
                #     cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
                #
                # cv2.namedWindow('img', 0)
                # mmcv.imshow(img, 'img')



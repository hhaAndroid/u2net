# 给定顶层文件夹，自动按照给定比例逐文件夹切分为train和test
import os
import numpy as np
import shutil


def copy_file(file_list, root_path, save_path, extensions, mask_extensions='_mask.png'):
    for file in file_list:
        _, filename = os.path.split(file)
        dst_file = os.path.join(save_path, filename)
        mask_path = filename.replace(extensions, mask_extensions)
        src_mask_path = os.path.join(root_path, mask_path)
        dst_mask_path = os.path.join(save_path, mask_path)
        shutil.copyfile(file, dst_file)
        if os.path.exists(src_mask_path):
            shutil.copyfile(src_mask_path, dst_mask_path)


def split_train_test_data(root_path, train_test_ratio, train_save_path, test_save_path, extensions='.png',
                          exclude_extensions='_mask.png'):
    if not os.path.exists(train_save_path):
        FileHelper.make_dirs(train_save_path)
    if not os.path.exists(test_save_path):
        FileHelper.make_dirs(test_save_path)
    paths = os.listdir(root_path)
    paths = map(lambda name: os.path.join(root_path, name), paths)
    for i in paths:
        root_p, filename = os.path.split(i)
        targetpath1 = os.path.join(train_save_path, filename)
        targetpath2 = os.path.join(test_save_path, filename)
        if FileHelper.is_dir(i):
            split_train_test_data(i, train_test_ratio, targetpath1, targetpath2)
        else:
            file_list = FileHelper.get_file_path_list(root_p, extensions, exclude_extensions)
            file_list = list(np.random.permutation(file_list).tolist())
            train_len = int(len(file_list) * train_test_ratio)
            copy_file(file_list[:train_len], root_p, train_save_path, extensions)
            copy_file(file_list[train_len:], root_p, test_save_path, extensions)
            break


def calc_datalen(root_path, extensions='.png', exclude_extensions=['_mask.png']):
    file_list = FileHelper.get_file_path_list(root_path, extensions, exclude_extensions)
    print(len(file_list))
    return len(file_list)


def cross_validation_data_split(root_path, save_path, Kfold):
    assert int(Kfold) > 1
    for i in range(int(Kfold) - 1):
        train_test_ratio = 1 - 1 / int(Kfold - i)
        if i == int(Kfold) - 2:
            train_save_path = os.path.join(save_path, 'train{}'.format(str(i + 1)))
        else:
            train_save_path = os.path.join(save_path, 'temp{}'.format(str(i)))
        test_save_path = os.path.join(save_path, 'train{}'.format(str(i)))
        split_train_test_data(root_path, train_test_ratio, train_save_path, test_save_path)
        root_path = train_save_path
        if i > 0:
            del_save_path = os.path.join(save_path, 'temp{}'.format(str(i - 1)))
            if os.path.exists(del_save_path):
                shutil.rmtree(del_save_path)


def demo_split_data_ratio(root_path):
    train_save_path = '/home/pi/dataset/defect_detection/PIE_data/train/'
    test_save_path = '/home/pi/dataset/defect_detection/PIE_data/test/'
    train_test_ratio = 0.6
    split_train_test_data(root_path, train_test_ratio, train_save_path, test_save_path)
    calc_datalen(train_save_path)
    calc_datalen(test_save_path)


if __name__ == '__main__':
    root_path = '/home/hha/dataset/circle/circle'
    calc_datalen(root_path)
    # 按照指定比例分成2份数据
    demo_split_data_ratio(root_path)

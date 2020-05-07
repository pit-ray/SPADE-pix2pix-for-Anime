#coding: utf-8
import os
import random
from glob import glob

import chainer.functions as F
import joblib
import numpy as np
from chainer.datasets.tuple_dataset import TupleDataset
from PIL import Image

from functions import label2onehot


def get_dataset(dir_path, augment=False, is_valid=True):
    os.makedirs('joblib', exist_ok=True)
    buf_file = 'joblib/' + dir_path.replace('/', '-') + '.job'

    if not os.path.exists(buf_file):
        x, t = [], []

        files = glob(dir_path + '/*.png')
        random.shuffle(files)
        for img_path in files:
            if not os.path.exists(img_path):
                continue

            print(img_path)
            img = Image.open(img_path)

            if img == None:
                continue

            img_array = np.array(img).astype('float16') / 255
            img_array = np.transpose(img_array, (2, 0, 1))

            t_array = img_array[:3, :, :256]
            x_array = img_array[:3, :, 256:512]
            c_array = img_array[:3, :, 512:]

            #to onehot
            x_array = label2onehot(x_array, threshold=0.4, skip_bg=True, dtype='float16')
            c_array = c_array * x_array[2]
            x_array = np.concatenate((x_array, c_array), axis=0)

            t_array = t_array * 2 - 1

            x.append(x_array)
            t.append(t_array)

            if augment:
                #mirroring
                x_mirror = x_array[:, :, ::-1]
                t_mirror = t_array[:, :, ::-1]
                x.append(x_mirror)
                t.append(t_mirror)

        with open(buf_file, 'wb') as f:
            joblib.dump((x, t), f, compress=3)

    else:
        with open(buf_file, 'rb') as f:
            x, t = joblib.load(f)


    if is_valid:
        train_size = int(len(x) * 0.9)
    else:
        train_size = len(x)

    train_x = x[:train_size]
    train_t = t[:train_size]
    valid_x = x[train_size:]
    valid_t = t[train_size:]

    return (TupleDataset(train_x, train_t), TupleDataset(valid_x, valid_t))

#coding: utf-8
import os
import random
import sys
from glob import glob

import chainer
import chainer.training.extensions as ex
import cupy as cp
import cv2
import numpy as np
from chainer import Variable, iterators, optimizer_hooks, optimizers, training
from chainer.backends import cuda
from PIL import Image

from dataset import get_dataset
from functions import label2onehot, onehot2label
from generator import SPADEGenerator
from options import get_options


def img_save(x, path):
    img_array = np.transpose(x, (1, 2, 0))
    img_array = np.uint8(img_array * 255)
    img = Image.fromarray(img_array)
    img.save(path)

def main():
    out_predict_dir = 'out'
    device = 0
    gen_npz = 'trained/gen_snapshot_epoch-900.npz'

    opt = get_options()

    opt.spade_ch = 32
    opt.ngf = 64
    opt.ndf = 64

    gen = SPADEGenerator(opt)
    gen.to_gpu(device)
    chainer.serializers.load_npz(gen_npz, gen)

    os.makedirs(out_predict_dir, exist_ok=True)

    out_dir = out_predict_dir + '/predicted'
    os.makedirs(out_dir, exist_ok=True)
    num = 0

    dir_path = 'datasets/resnet-large_hc'
    files = glob(dir_path + '/*.png')
    random.shuffle(files)

    for img_path in files:
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path)

        if img == None:
            continue
        print(img_path)

        img_array = np.array(img).astype('float32') / 255
        img_array = np.transpose(img_array, (2, 0, 1))

        t_array = img_array[:3, :, :256]
        x_array = img_array[:3, :, 256:512]
        c_array = img_array[:3, :, 512:]

        #to onehot
        x_array = label2onehot(x_array, threshold=0.4, skip_bg=True, dtype='float32').astype('float32')
        c_array = c_array * x_array[2]

        #cast 16bit -> 32bit (cannot use tensor core)
        x = Variable(cuda.to_gpu(x_array[np.newaxis, :, :, :]))
        c = Variable(cuda.to_gpu(c_array[np.newaxis, :, :, :]))

        out = gen([x, c])[0]

        out = cp.asnumpy(out.array[0])
        out= (out + 1) / 2
        x = cp.asnumpy(x.array[0])
        x = onehot2label(x, skip_bg=True, dtype='float32')

        out = np.transpose(out * 255, (1, 2, 0)).astype('uint8')
        x = np.transpose(x * 255, (1, 2, 0)).astype('uint8')

        y = np.transpose(t_array * 255, (1, 2, 0)).astype('uint8')

        out_img = np.concatenate((x, y, out), axis=1)
        img = Image.fromarray(out_img)
        path = out_dir + '/' + str(num) + '.png'
        img.save(path)

        num += 1

if __name__ == '__main__':
    main()

# coding: utf-8
import os

import chainer
import numpy as np
from PIL import Image
from glob import glob

from functions import label2onehot
from generator import SPADEGenerator
from options import get_options


def main():
    out_dir = 'predict_to'
    in_dir = 'predict_from'
    gen_npz = 'pretrained/gen.npz'

    opt = get_options()

    gen = SPADEGenerator(opt)
    gen.to_gpu(0)
    chainer.serializers.load_npz(gen_npz, gen)
    gen.to_cpu()

    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    files = glob(in_dir + '/*.*')
    if len(files) == 0:
        print('Erorr: No files to load in \'' + in_dir + '\'.')
        return

    num = 0
    for filename in files:
        print(filename + ': ', end="")
        src_img = Image.open(filename).convert('RGB')
        if src_img is None:
            print('Not Loaded')
            continue

        print('Loaded')
        src_array = np.array(src_img, dtype='float32')
        src_array = src_array.transpose((2, 0, 1)) / 255

        x_array = src_array[:3, :, :256]
        c_array = src_array[:3, :, 256:512]

        x_onehot = label2onehot(x_array, threshold=0.4, skip_bg=True, dtype='float32')
        x = chainer.Variable(x_onehot[np.newaxis, :, :, :].astype('float32'))

        c_array = c_array * x_onehot[2]  # crop with hair label
        c = chainer.Variable(c_array[np.newaxis, :, :, :].astype('float32'))

        out = gen([x, c])

        x_array = np.transpose(x_array, (1, 2, 0))
        out_array = np.transpose((out.array[0] + 1) / 2, (1, 2, 0))

        img_array = np.concatenate((x_array, out_array), axis=1) * 255
        img = Image.fromarray(img_array.astype('uint8'))

        path = out_dir + '/' + str(num) + '.png'
        img.save(path)

        num += 1


if __name__ == '__main__':
    main()

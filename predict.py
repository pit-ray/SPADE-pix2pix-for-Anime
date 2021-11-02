# coding: utf-8
import os

import tqdm

import chainer
import numpy as np
from PIL import Image
from glob import glob

from functions import label2onehot
from generator import SPADEGenerator
from options import get_parser


def predict(opt):
    gen_npz = 'pretrained/gen.npz'

    gen = SPADEGenerator(opt)
    gen.to_gpu(0)
    chainer.serializers.load_npz(gen_npz, gen)
    gen.to_cpu()

    os.makedirs(opt.srcdir, exist_ok=True)
    os.makedirs(opt.dstdir, exist_ok=True)

    files = glob(os.path.join(opt.srcdir, '*.*'))
    if len(files) == 0:
        print('Erorr: No files to load in \'' + opt.dstdir + '\'.')
        return

    with tqdm.tqdm(files, leave=False) as pbar:
        for filename in pbar:
            src_img = Image.open(filename).convert('RGB')
            if src_img is None:
                continue

            pbar.set_postfix({'input': filename})

            src_array = np.array(src_img, dtype='float32')
            src_array = src_array.transpose((2, 0, 1)) / 255

            C, H, W = src_array.shape
            if W == H * 2:
                x_array = src_array[:3, :, :256]
                c_array = src_array[:3, :, 256:512]
            else:
                x_array = src_array[:3, :, :256]
                rgb = np.random.rand(3, 1, 1)
                c_array = rgb.repeat(256, axis=1).repeat(256, axis=2)

            x_onehot = label2onehot(x_array, threshold=0.4, skip_bg=True, dtype='float32')
            x = chainer.Variable(x_onehot[np.newaxis, :, :, :].astype('float32'))

            c_array = c_array * x_onehot[2]  # crop with hair label
            c = chainer.Variable(c_array[np.newaxis, :, :, :].astype('float32'))

            out = gen([x, c])

            x_array = np.transpose(x_array, (1, 2, 0))
            out_array = np.transpose((out.array[0] + 1) / 2, (1, 2, 0))

            img_array = np.concatenate((x_array, out_array), axis=1) * 255
            img = Image.fromarray(img_array.astype('uint8'))

            out_path = os.path.join(opt.dstdir, os.path.basename(filename))
            img.save(out_path)


if __name__ == '__main__':
    parser = get_parser()

    parser.add_argument('--srcdir', type=str, default='predict_from')
    parser.add_argument('--dstdir', type=str, default='predict_to')

    opt = parser.parse_args()

    predict(opt)

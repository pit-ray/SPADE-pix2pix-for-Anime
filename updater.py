#coding: utf-8
import os

import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Variable, using_config
from chainer.backends import cuda
from chainer.training.updaters import StandardUpdater
from PIL import Image

from functions import onehot2label
from loss import dis_loss, gen_loss
from perceptual_loss import PerceptualLoss


class pix2pix_Updater(StandardUpdater):
    def __init__(self, opt, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.opt = opt
        self.num_saved_img = 0
        self.perc_func = PerceptualLoss(opt)
        self.img4save = []

    def update_core(self):
        for _ in range(self.opt.n_dpg):
            self.update_D(self.get_iterator('main'))

        self.update_G(self.get_iterator('main'))

        if not self.opt.use_rough:
            return

        rough_iter = self.get_iterator('rough')

        for _ in range(self.opt.n_dpg):
            self.update_D(rough_iter, is_print=False, loss_weight=self.opt.rough_loss_weight)

        self.update_G(rough_iter, is_print=False, loss_weight=self.opt.rough_loss_weight)

    def update_D(self, train_iter, is_print=True, loss_weight=1):
        gen = self.get_optimizer('gen').target
        dis_opt = self.get_optimizer('dis')
        dis = dis_opt.target

        label, real_g, condition = self.real_batch(train_iter)
        g_out = gen([label, condition])
        fake_g = g_out

        real_d = dis([label, real_g, condition])[0]
        fake_d = dis([label, fake_g, condition])[0]

        d_loss = dis_loss(self.opt, real_d, fake_d, real_g, fake_g, observer=dis if is_print else None)
        d_loss *= loss_weight

        fake_g.unchain_backward()

        dis.cleargrads()
        d_loss.backward()
        dis_opt.update()

    def update_G(self, train_iter, is_print=True, loss_weight=1):
        gen_opt = self.get_optimizer('gen')
        gen = gen_opt.target
        dis = self.get_optimizer('dis').target

        label, real_g, condition = self.real_batch(train_iter)
        fake_g = gen([label, condition])

        if is_print:
            self.img4save = [cuda.to_cpu(label.array[0]),
                             cuda.to_cpu(real_g.array[0]),
                             cuda.to_cpu(fake_g.array[0])]

        real_d, real_d_fm = dis([label, real_g, condition])
        fake_d, fake_d_fm = dis([label, fake_g, condition])

        g_loss = gen_loss(self.opt, fake_d, real_g, fake_g, real_d_fm, fake_d_fm,
            perceptual_func=self.perc_func, observer=gen if is_print else None)

        g_loss *= loss_weight

        gen.cleargrads()
        g_loss.backward()
        gen_opt.update()

    def real_batch(self, iterator):
        batch = iterator.next()
        x, y = self.converter(batch, self.device)

        c = x[:, -3:, :, :]
        x = x[:, :-3, :, :]

        #cast 16bit -> 32bit (cannot use tensor core)
        x = Variable(x.astype('float32'))
        y = Variable(y.astype('float32'))
        c = Variable(c.astype('float32'))

        return x, y, c

    def evaluate(self):
        with using_config('train', False):
            with using_config('enable_backprop', False):
                gen = self.get_optimizer('gen').target
                dis = self.get_optimizer('dis').target

                label, real_g, condition = self.real_batch(self.get_iterator('valid'))
                fake_g = gen([label, condition])

                to_save_img = [cuda.to_cpu(label.array[0]),
                            cuda.to_cpu(real_g.array[0]),
                            cuda.to_cpu(fake_g.array[0])]

            return to_save_img

            #real_d, real_d_fm = dis([label, real_g, condition])
            #fake_d, fake_d_fm = dis([label, fake_g, condition])

            #dis_loss(self.opt, real_d, fake_d, real_g, fake_g,
            #    observer=dis, tag='valid')

            #gen_loss(self.opt, fake_d, real_g, fake_g, real_d_fm, fake_d_fm,
            #    perceptual_func=self.perc_func, observer=gen, tag='valid')

            #return to_save_img

    def save_img(self):
        out_dir = self.opt.out_dir + '/img/'
        os.makedirs(out_dir, exist_ok=True)
        train = self.img4save
        valid1 = self.evaluate()
        valid2 = self.evaluate()
        
        tile_img = None
        for seed, real, fake in [train, valid1, valid2]:
            seed = onehot2label(seed, skip_bg=True, dtype='float32')

            tp = (1, 2, 0)
            seed = np.transpose(seed, tp)
            real = np.transpose(real, tp)
            fake = np.transpose(fake, tp)

            real = (real + 1) / 2
            fake = (fake + 1) / 2

            row_img = np.concatenate((seed, real, fake), axis=1)

            if tile_img is None:
                tile_img = row_img
            else:
                tile_img = np.concatenate((tile_img, row_img), axis=0)

        img_array = np.uint8(tile_img * 255)
        img = Image.fromarray(img_array)
        img.save(out_dir + 'tile_img-' + str(self.num_saved_img) + '.png')
        self.num_saved_img += 1

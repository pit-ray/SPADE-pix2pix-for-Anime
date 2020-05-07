#coding: utf-8
import chainer.functions as F
import chainer.links as L
from chainer import Chain, Sequential, Variable
from chainer.backends import cuda
from chainer.initializers import HeNormal, Normal

from architectures import ConstantFCN, SPADEResBlock, define_upsampling
from spectral_norm import define_conv


class SPADEGenerator(Chain):
    def __init__(self, opt):
        super().__init__()

        xavier_w = Normal()
        he_w = HeNormal()

        C, H, W = opt.img_shape
        ngf = opt.ngf
        label_ch = opt.class_num
        self.opt = opt

        layer_num = 6
        init_shape = (ngf * 8, H // 2 ** layer_num, W // 2 ** layer_num)

        with self.init_scope(): 
            self.w1 = ConstantFCN(opt, opt.class_num)
            self.w2 = ConstantFCN(opt, opt.c_shape[0])

            self.head_reshape = lambda x: F.resize_images(x, init_shape[1:])
            self.head = define_conv(opt)(label_ch, init_shape[0], ksize=3, pad=1, initialW=he_w)

            #512 x 4 x 4
            self.r1 = SPADEResBlock(opt, ngf * 8, ngf * 8)
            self.up1 = define_upsampling(opt, ngf * 8)

            #512 x 8 x 8
            self.r2 = SPADEResBlock(opt, ngf * 8, ngf * 8)
            self.up2 = define_upsampling(opt, ngf * 8)

            #512 x 16 x 16
            self.r3 = SPADEResBlock(opt, ngf * 8, ngf * 8)
            self.up3 = define_upsampling(opt, ngf * 8)

            #512 x 32 x 32
            self.r4 = SPADEResBlock(opt, ngf * 8, ngf * 4)
            self.up4 = define_upsampling(opt, ngf * 4)

            #256 x 64 x 64
            self.r5 = SPADEResBlock(opt, ngf * 4, ngf * 2)
            self.up5 = define_upsampling(opt, ngf * 2)

            #128 x 128 x 128
            self.r6 = SPADEResBlock(opt, ngf * 2, ngf)
            self.up6 = define_upsampling(opt, ngf)

            #64 x 256 x 256
            self.r7 = SPADEResBlock(opt, ngf, ngf // 2)

            #32 x 256 x 256
            self.to_img = L.Convolution2D(ngf // 2, 3, ksize=3, pad=1, initialW=xavier_w)

    def __call__(self, inputs):
        label, condition = inputs

        w1 = self.w1(label)
        w2 = self.w2(condition)

        #constant input
        h = self.head_reshape(label)
        h = self.head(h)

        h = self.r1(h, w1)
        h = self.up1(h)

        #8
        h = self.r2(h, w1)
        h = self.up2(h)

        #16
        h = self.r3(h, w1)
        h = self.up3(h)

        #32
        h = self.r4(h, w1)
        h = self.up4(h)

        #64
        h = self.r5(h, w1)
        h = self.up5(h)

        #128
        h = self.r6(h, w1, label2=w2)
        h = self.up6(h)

        #256
        h = self.r7(h, w1, label2=w2)
        h = F.leaky_relu(h)

        h = self.to_img(h)
        out = F.tanh(h)

        return out

#coding: utf-8
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer.initializers import HeNormal, Normal

from architectures import ResBlock, SelfAttention, define_upsampling
from atrous_conv import define_atrous_conv
from spectral_norm import define_conv, define_deconv


class PatchDiscriminator(Chain):
    def __init__(self, opt):
        super().__init__()

        he_w = HeNormal()
        xavier_w = Normal()
        ndf = opt.ndf
        with self.init_scope():
            top_ch = opt.img_shape[0] + opt.class_num + opt.c_shape[0]

            #top_ch x 256 x 256
            self.c1 = define_conv(opt)(top_ch, ndf, ksize=4, stride=2, pad=1, initialW=he_w)

            #64 x 128 x 128
            self.c2 = define_conv(opt)(ndf, ndf * 2, ksize=4, stride=2, pad=1, initialW=he_w)
            self.n2 = L.BatchNormalization(size=ndf * 2)

            #128 x 64 x 64
            self.c3 = define_conv(opt)(ndf * 2, ndf * 4, ksize=4, stride=2, pad=1, initialW=he_w)
            self.att = SelfAttention(opt, ndf * 4)
            self.n3 = L.BatchNormalization(size=(ndf * 4))

            #256 x 32 x 32
            self.c4 = define_conv(opt)(ndf * 4, ndf * 8, ksize=4, stride=1, pad=1, initialW=he_w)
            self.n4 = L.BatchNormalization(size=ndf * 8)

            #512 x 31 x 32
            self.head = define_conv(opt)(ndf * 8, ndf * 8, ksize=4, stride=1, pad=1, initialW=he_w)

            #512 x 30 x 30
            self.r1 = ResBlock(opt, ndf * 8, ndf * 4)

            #256 x 30 x 30
            self.r2 = ResBlock(opt, ndf * 4, ndf * 2)

            #128 x 30 x 30
            self.r3 = ResBlock(opt, ndf * 2, ndf)

            #64 x 30 x 30
            self.to_patch = define_conv(opt)(ndf, 1, ksize=3, stride=1, pad=1, initialW=xavier_w)

            #out is 1 x 30 x 30

            self.activation = F.leaky_relu

    def __call__(self, inputs):
        fm = []
        h = F.concat(inputs, axis=1)

        h = self.c1(h)
        h = self.activation(h)
        fm.append(h)

        h = self.c2(h)
        h = self.n2(h)
        h = self.activation(h)
        fm.append(h)

        h = self.c3(h)
        h = self.att(h)
        h = self.n3(h)
        h = self.activation(h)
        fm.append(h)

        h = self.c4(h)
        h = self.n4(h)
        h = self.activation(h)
        fm.append(h)

        h = self.head(h)

        h = self.r1(h)
        h = self.r2(h)
        h = self.r3(h)

        h = self.activation(h)
        fm.append(h)

        out = self.to_patch(h)
        return (out, fm)
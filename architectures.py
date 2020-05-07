#coding: utf-8
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Parameter, Sequential, Variable
from chainer.backends import cuda
from chainer.initializers import HeNormal

from atrous_conv import define_atrous_conv
from spectral_norm import define_conv, define_deconv


class SPADE(Chain):
    def __init__(self, opt, ch):
        super().__init__()

        he_w = HeNormal()

        with self.init_scope():
            #not affine normalization
            self.norm = L.BatchNormalization(size=ch, use_gamma=False, use_beta=False)

            self.gamma = L.Convolution2D(opt.spade_ch, ch, ksize=3, pad=1, initialW=he_w)
            self.beta = L.Convolution2D(opt.spade_ch, ch, ksize=3, pad=1, initialW=he_w)

            self.activation = F.leaky_relu

    def __call__(self, x, label):
        normed = self.norm(x)

        resized = F.resize_images(label, x.shape[2:])

        gamma = self.gamma(resized)
        beta = self.beta(resized)

        out = normed * (1 + gamma) + beta
        return out


class SelfAttention(Chain):
    #Self Attention GAN v2
    def __init__(self, opt, ch):
        super().__init__()

        he_w = HeNormal()
        mid_ch = ch // opt.division_ch

        with self.init_scope():
            self.f_conv = define_conv(opt)(ch, mid_ch, ksize=1, initialW=he_w)
            self.g_conv = define_conv(opt)(ch, mid_ch, ksize=1, initialW=he_w)
            self.h_conv = define_conv(opt)(ch, mid_ch, ksize=1, initialW=he_w)
            self.v_conv = define_conv(opt)(mid_ch, ch, ksize=1, initialW=he_w)
            self.gamma = Parameter(initializer=0, shape=1, name='SA-gamma')

    def __call__(self, x):
        query = self.f_conv(x)
        key = self.g_conv(x)

        #compute vertival and horizontal pixel information
        qB, qC, qW, qH = query.shape
        query = F.reshape(query, (qB, qC, qW * qH))

        kB, kC, kW, kH = key.shape
        key = F.reshape(key, (kB, kC, kW * kH))

        #compute matrix product vertical pixel information and horizontal,
        #its softmax has resemble feature quantity(this is attention map).
        s_map = F.matmul(query, key, transa=True)
        a_map = F.softmax(s_map, axis=1)

        weight = self.h_conv(x)
        wB, wC, wW, wH = weight.shape
        weight = F.reshape(weight, (wB, wC, wW * wH))

        #attention to resemble feature by producting attention map and original map
        sa_fmap = F.matmul(weight, a_map)
        sa_fmap = F.reshape(sa_fmap, (wB, wC, kW, kH))
        sa_fmap = self.v_conv(sa_fmap)

        out = self.gamma*sa_fmap + x
        return out


class ConstantFCN(Chain):
    def __init__(self, opt, input_ch):
        super().__init__()

        he_w = HeNormal()
        ch = opt.spade_ch

        with self.init_scope():
            #256
            self.conv1 = L.Convolution2D(input_ch, ch // 8, ksize=3, pad=1, initialW=he_w)
            #128
            self.conv2 = L.Convolution2D(ch // 8, ch // 4, ksize=3, pad=1, initialW=he_w)
            #64
            self.conv3 = L.Convolution2D(ch // 4, ch // 2, ksize=3, pad=2, initialW=he_w, dilate=2)
            #32
            self.conv4 = L.Convolution2D(ch // 2, ch, ksize=3, pad=4, initialW=he_w, dilate=4)
            #16

            self.activation = F.leaky_relu

    def __call__(self, x):
        h = self.conv1(x)
        h = self.activation(h)

        h = self.conv2(h)
        h = self.activation(h)

        h = self.conv3(h)
        h = self.activation(h)

        h = self.conv4(h)
        h = self.activation(h)

        return h


class ASPP(Chain):
    def __init__(self, opt, input_ch, input_resolution=65):
        super().__init__()

        #get options
        #nf = opt.aspp_nf
        nf = 128

        #this rate is dilate size based original paper.
        x65_rate = [6, 12, 18]
        rate = [round(x * input_resolution / 65) for x in x65_rate]

        he_w = HeNormal()
        with self.init_scope():
            self.x1 = define_conv(opt)(input_ch, nf, ksize=1, initialW=he_w)
            self.x1_bn = L.BatchNormalization(nf)

            self.x3_small = define_atrous_conv(opt)(input_ch, nf, ksize=3, rate=rate[0], initialW=he_w)
            self.x3_small_bn = L.BatchNormalization(nf)
            
            self.x3_middle = define_atrous_conv(opt)(input_ch, nf, ksize=3, rate=rate[1], initialW=he_w)
            self.x3_middle_bn = L.BatchNormalization(nf)

            self.x3_large = define_atrous_conv(opt)(input_ch, nf, ksize=3, rate=rate[2], initialW=he_w)
            self.x3_large_bn = L.BatchNormalization(nf)

            self.sum_func = define_conv(opt)(nf * 4, input_ch, ksize=3, pad=1, initialW=he_w)

        self.activation = F.leaky_relu

    def __call__(self, x):
        h1 = self.x1(x)
        h1 = self.x1_bn(h1)
        h1 = self.activation(h1)

        h2 = self.x3_small(x)
        h2 = self.x3_small_bn(h2)
        h2 = self.activation(h2)

        h3 = self.x3_middle(x)
        h3 = self.x3_middle_bn(h3)
        h3 = self.activation(h3)

        h4 = self.x3_large(x)
        h4 = self.x3_large_bn(h4)
        h4 = self.activation(h4)

        out = F.concat((h1, h2, h3, h4), axis=1)
        out = self.sum_func(out)
        out = self.activation(out)

        return out


class NoiseAdder(Chain):
    def __init__(self, ch):
        super().__init__()

        with self.init_scope():
            self.gamma = Parameter(initializer=0, shape=(1, ch, 1, 1), name='noise_gamma')

    def __call__(self, x, mean=None, ln_var=None):
        xp = cuda.get_array_module(x.array)
        if mean is None and ln_var is None:
            noise = xp.random.normal(size=(x.shape[0], 1, x.shape[2], x.shape[3]), dtype='float32')
            noise = Variable(noise)
        else:
            noise = F.gaussian(mean, ln_var)

        out = x + self.gamma * noise
        return out


class ResBlock(Chain):
    def __init__(self, opt, in_ch, out_ch, out_conv_initW=HeNormal()):
        super().__init__()

        he_w = HeNormal()
        with self.init_scope():
            self.norm1 = L.BatchNormalization(size=in_ch)
            self.conv1 = define_conv(opt)(in_ch, out_ch, ksize=3, pad=1, initialW=he_w)

            self.norm2 = L.BatchNormalization(size=out_ch)
            self.conv2 = define_conv(opt)(out_ch, out_ch, ksize=3, pad=1, initialW=out_conv_initW)

            self.activation = F.leaky_relu

            #if input channel is not equel with output channel, input channel convert to output shape
            if in_ch != out_ch:
                self.reshape_norm = L.BatchNormalization(size=in_ch)
                self.reshape_act = self.activation
                self.reshape_conv = define_conv(opt)(in_ch, out_ch, ksize=3, pad=1, initialW=out_conv_initW)
            else:
                self.reshape_norm = lambda x: x
                self.reshape_act = lambda x: x
                self.reshape_conv = lambda x: x

    def __call__(self, x):
        rh = self.reshape_norm(x)
        rh = self.reshape_act(rh) #in paper, this activation is exist, however, original github is not.
        rh = self.reshape_conv(rh)

        h = self.norm1(x)
        h = F.leaky_relu(h)
        h = self.activation(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)

        out = h + rh

        return out


class SPADEResBlock(Chain):
    def __init__(self, opt, in_ch, out_ch, out_conv_initW=HeNormal()):
        super().__init__()

        he_w = HeNormal()
        with self.init_scope():
            self.noise1 = NoiseAdder(in_ch)
            self.norm1 = SPADE(opt, in_ch)
            self.conv1 = define_conv(opt)(in_ch, out_ch, ksize=3, pad=1, initialW=he_w)

            self.noise2 = NoiseAdder(out_ch)
            self.norm2 = SPADE(opt, out_ch)
            self.conv2 = define_conv(opt)(out_ch, out_ch, ksize=3, pad=1, initialW=out_conv_initW)

            self.activation = F.leaky_relu

            #if input channel is not equel with output channel,
            #input channel convert to output shape
            if in_ch != out_ch:
                self.reshape_noise = NoiseAdder(in_ch)
                self.reshape_norm = SPADE(opt, in_ch)
                self.reshape_act = self.activation
                self.reshape_conv = define_conv(opt)(in_ch, out_ch, ksize=1, initialW=out_conv_initW, nobias=True)

            else:
                self.reshape_noise = lambda x, mean=None, ln_var=None : x
                self.reshape_norm = lambda x, y: x
                self.reshape_act = lambda x: x
                self.reshape_conv = lambda x: x

    def __call__(self, x, label, label2=None, mean=None, ln_var=None):
        rh = self.reshape_noise(x, mean=mean, ln_var=ln_var)
        rh = self.reshape_norm(rh, label)
        rh = self.reshape_act(rh)
        rh = self.reshape_conv(rh)

        w = label2 if label2 is not None else label
        h = self.noise1(x, mean=mean, ln_var=ln_var)
        h = self.norm1(h, w)
        h = self.activation(h)
        h = self.conv1(h)

        h = self.noise2(h, mean=mean, ln_var=ln_var)
        h = self.norm2(h, label)
        h = self.activation(h)
        h = self.conv2(h)

        out = h + rh
        return out


class PixelShuffler(Chain):
    def __init__(self, opt, input_ch, output_ch=None, rate=2):
        super().__init__()
        he_w = HeNormal()

        if output_ch is None:
            output_ch = input_ch

        output_ch = output_ch * rate**2

        with self.init_scope():
            self.c = define_conv(opt)(input_ch, output_ch, ksize=3, stride=1, pad=1, initialW=he_w)

        self.ps_func = lambda x: F.depth2space(x, rate)

    def __call__(self, x):
        out = self.c(x)
        out = self.ps_func(out)

        return out


def define_upsampling(opt, input_ch, output_ch=None):
    if opt.upsampling_mode == 'bilinear':
        seq = Sequential(lambda x: F.resize_images(x, (x.shape[2] * 2, x.shape[3] * 2), mode='bilinear'))

        if output_ch is not None:
            seq.append(define_conv(opt)(input_ch, output_ch, ksize=3, stride=1, pad=1, initialW=HeNormal()))

        return seq
    
    if opt.upsampling_mode == 'nearest':
        seq =  Sequential(lambda x: F.resize_images(x, (x.shape[2] * 2, x.shape[3] * 2), mode='nearest'))

        if output_ch is not None:
            seq.append(define_conv(opt)(input_ch, output_ch, ksize=3, stride=1, pad=1, initialW=HeNormal()))

        return seq

    if opt.upsampling_mode == 'deconv':
        return define_deconv(opt)(input_ch, input_ch if output_ch is None else output_ch,
            ksize=3, stride=1, pad=1, initialW=HeNormal())

    if opt.upsampling_mode == 'subpx_conv':
        return PixelShuffler(opt, input_ch, input_ch if output_ch is None else output_ch)

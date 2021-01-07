#coding: utf-8
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer.backends import cuda


class PerceptualLoss(Chain):
    def __init__(self, opt):
        super().__init__()
        with self.init_scope():
            self.detecter = L.VGG16Layers().to_gpu(0)
            self.layer_names = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']

            if opt.perceptual_model == 'VGG19':
                self.detecter = L.VGG19Layers().to_gpu(0)
                self.layer_name = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4']

        self.weight = [32 ** -1,
                       16 ** -1,
                       8 ** -1,
                       4 ** -1,
                       1]

        self.coef = opt.perceptual_coef
        self.criterion = F.mean_absolute_error

        if opt.perceptual_mode == 'MAE':
            self.criterion = F.mean_absolute_error

        if opt.perceptual_mode == 'MSE':
            self.criterion = F.mean_squared_error
            self.coef *= 0.5

    def prepare(self, variable_img):
        #out = F.resize_images(variable_img, (224, 224))
        out = variable_img
        out = (out + 1) / 2
        out = out[:, ::-1, :, :]
        out = F.transpose(out, (0, 2, 3, 1))

        out *= 255
        xp = cuda.get_array_module(variable_img.array)
        out -= xp.array([103.939, 116.779, 123.68], dtype=variable_img.dtype)

        out = F.transpose(out, (0, 3, 1, 2))

        return out

    def __call__(self, real, fake):
        loss = 0
        real = self.prepare(real)
        fake = self.prepare(fake)

        real_feat = self.detecter(real, layers=self.layer_names)
        fake_feat = self.detecter(fake, layers=self.layer_names)

        for i, name in enumerate(self.layer_names):
            loss += self.weight[i] * self.criterion(real_feat[name].array, fake_feat[name]) / 255

        loss *= self.coef
        return loss
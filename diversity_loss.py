#coding: utf-8
import chainer.functions as F

def diversity_loss(self, z_variable, g_x, label, color_map, real_y, coef=1):
    #target sample
    #saved memory

    sample_fake_y, _, _, sample_z = self.fake_batch(label, real_y, color_map=color_map)

    #only specific class
    g_x *= label[:, 3, :, :]
    sample_fake_y *= label[:, 3, :, :]

    eps = 1e-5
    #compute distance
    z_dis = F.mean_absolute_error(z_variable, sample_z.array)
    img_dis = F.mean_absolute_error(sample_fake_y.array, g_x)

    diversity_ratio = img_dis / z_dis
    loss = coef * 1 / (diversity_ratio + eps)

    return loss, diversity_ratio
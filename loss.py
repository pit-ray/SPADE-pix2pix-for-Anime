#coding: utf-8
import chainer.functions as F
from chainer import Variable, grad, report
from chainer.backends import cuda


def compute_grad(d_out, x):
    d_out_sum = F.mean(d_out, axis=(1, 2, 3))
    gradient = grad([d_out_sum], [x], enable_double_backprop=True)[0]
    gradient = F.sum(gradient, axis=(1, 2, 3))
    out = gradient**2
    return out


def l2f(loss):
    if isinstance(loss, Variable):
        loss = loss.array

    return float(loss)


def dis_loss(opt, real_d, fake_d, real_g, fake_g, observer=None, tag=str()):
    #gradient penalty
    real_gp = 0
    fake_gp = 0
    if opt.zero_gp_mode == 'real' or opt.zero_gp_mode == 'real_fake':
        real_gp = opt.gp_coef * compute_grad(real_d, real_g)

    if opt.zero_gp_mode == 'fake' or opt.zero_gp_mode == 'real_fake':
        fake_gp = opt.gp_coef * compute_grad(fake_d, fake_g)

    #adversarial loss
    adv_loss = 0
    real_loss = 0
    fake_loss = 0
    if opt.adv_loss_mode == 'wgan':
        adv_loss = -F.mean(real_d - fake_d)
        gp = real_gp + fake_gp

    else:
        if opt.adv_loss_mode == 'bce':
            real_loss = F.mean(F.softplus(-real_d))
            fake_loss = F.mean(F.softplus(fake_d))

        if opt.adv_loss_mode == 'mse':
            xp = cuda.get_array_module(real_d.array)
            real_loss = F.mean_squared_error(real_d, xp.ones_like(real_d.array))
            fake_loss = F.mean_squared_error(fake_d, xp.zeros_like(fake_d.array))

        if opt.adv_loss_mode == 'hinge':
            real_loss = F.mean(F.relu(1.0 - real_d))
            fake_loss = F.mean(F.relu(1.0 + fake_d))

        adv_loss = (real_loss + fake_loss) * 0.5
        gp = (real_gp + fake_gp) * 0.5

    loss = adv_loss + gp

    if observer is not None:
        if tag:
            tag += '_'

        report({tag + 'loss': l2f(loss),
                tag + 'adv_loss': l2f(adv_loss),
                tag + 'real_loss': l2f(real_loss),
                tag + 'fake_loss': l2f(fake_loss),
                tag + 'gp': l2f(gp),
                tag + 'adv_loss_with_gp': l2f(adv_loss + gp)}, observer=observer)

    return loss


def gen_loss(opt, fake_d, real_g, fake_g, real_d_fm, fake_d_fm, perceptual_func=None, observer=None, tag=str()):
    #adversarial loss
    adv_loss = 0
    fake_loss = 0
    if opt.adv_loss_mode == 'bce':
        fake_loss = F.mean(F.softplus(-fake_d))

    if opt.adv_loss_mode == 'mse':
        xp = cuda.get_array_module(fake_d.array)
        fake_loss = F.mean_squared_error(fake_d, xp.ones_like(fake_d.array))

    if opt.adv_loss_mode == 'hinge':
        fake_loss = -F.mean(fake_d)

    if opt.adv_loss_mode == 'wgan':
        fake_loss += -F.mean(fake_d)

    adv_loss = fake_loss

    fm_loss = 0
    #feature matching loss
    if opt.fm_coef != 0:
        layer_num = len(fake_d_fm)
        for rfm, ffm in zip(real_d_fm, fake_d_fm):
            fm_loss += opt.fm_coef * F.mean_absolute_error(rfm.array, ffm) / layer_num

    perceptual_loss = 0
    if perceptual_func is not None:
        perceptual_loss += perceptual_func(real_g, fake_g)

    loss = adv_loss + fm_loss + perceptual_loss

    if observer is not None:
        if tag:
            tag += '_'

        report({tag + 'loss': l2f(loss),
                tag + 'adv_loss': l2f(adv_loss),
                tag + 'fake_loss': l2f(fake_loss),
                tag + 'fm_loss': l2f(fm_loss),
                tag + 'perceptual_loss': l2f(perceptual_loss)}, observer=observer)

    return loss
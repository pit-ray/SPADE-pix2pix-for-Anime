#coding: utf-8
import chainer.training.extensions as ex
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam
from chainer.serializers import load_npz
from chainer.training import PRIORITY_READER, Trainer

from dataset import get_dataset
from discriminator import PatchDiscriminator
from functions import adam_lr_poly
from generator import SPADEGenerator
from options import get_options
from updater import pix2pix_Updater


def train(opt):
    if opt.use_cpu:
        device = -1
        print('[Msg] use CPU')
    else:
        device = 0
        print('[Msg] use GPU0')

    train, valid = get_dataset(opt.dataset_dir, augment=True)
    print('[Msg] loaded normal dataset')
    if opt.use_rough:
        rough_train, _ = get_dataset(opt.rough_dataset_dir, augment=False, is_valid=False)
        print('[Msg] loaded rough dataset')

    train_iter = SerialIterator(train, opt.batch_size, shuffle=True, repeat=True)
    valid_iter = SerialIterator(valid, opt.batch_size, shuffle=True, repeat=True)

    if opt.use_rough:
        rough_iter = SerialIterator(rough_train, opt.batch_size, shuffle=True, repeat=True)
    else:
        rough_iter = train_iter
    print('[Msg] convert dataset to iterator')

    gen = SPADEGenerator(opt)
    if device != -1:
        gen.to_gpu(device) #use GPU
    if opt.dis_snapshot:
        load_npz(opt.gen_snapshot, gen, strict=False)
        print('[Msg] loaded gen npz (' + opt.gen_snapshot + ')')
    g_optim = Adam(alpha=opt.g_lr, beta1=opt.g_beta1, beta2=opt.g_beta2)
    g_optim.setup(gen)
    print('[Msg] completed generator setup')

    dis = PatchDiscriminator(opt)
    if device != -1:
        dis.to_gpu(device) #use GPU
    if opt.dis_snapshot:
        load_npz(opt.dis_snapshot, dis, strict=False)
        print('[Msg] loaded dis npz (' + opt.dis_snapshot + ')')
    d_optim = Adam(alpha=opt.d_lr, beta1=opt.d_beta1, beta2=opt.d_beta2)
    d_optim.setup(dis)
    print('[Msg] completed discriminator setup')

    updater = pix2pix_Updater(opt,
        iterator={'main': train_iter, 'valid': valid_iter, 'rough': rough_iter},
        optimizer={'gen': g_optim, 'dis': d_optim},
        device=device)

    trainer = Trainer(updater, (opt.max_iteration, 'iteration'), out=opt.out_dir)
    print('[Msg] created updater and trainer')

    #chainer training extensions
    trainer.extend(ex.LogReport(log_name=None, trigger=(1, 'iteration')))
    trainer.extend(ex.ProgressBar((opt.max_iteration, 'iteration'), update_interval=1))

    #plot
    #adv loss
    trainer.extend(ex.PlotReport(['gen/adv_loss', 'dis/adv_loss'],
        x_key='iteration', filename='adv-loss.png', trigger=(25, 'iteration')))

    trainer.extend(ex.PlotReport(['gen/adv_loss', 'dis/adv_loss'],
        x_key='epoch', filename='adv-loss(details).png', trigger=(2, 'epoch')))

    #adv loss with gp
    trainer.extend(ex.PlotReport(['gen/adv_loss', 'dis/adv_loss_with_gp'],
        x_key='iteration', filename='adv-loss-with-gp.png', trigger=(25, 'iteration')))

    trainer.extend(ex.PlotReport(['gen/adv_loss', 'dis/adv_loss_with_gp'],
        x_key='epoch', filename='adv-loss-with-gp(details).png', trigger=(2, 'epoch')))

    trainer.extend(ex.PlotReport(['gen/adv_loss', 'dis/adv_loss_with_gp', 'gen/valid_adv_loss', 'dis/valid_adv_loss_with_gp'],
        x_key='epoch', filename='adv-loss-with-gp-and-valid.png', trigger=(2, 'epoch')))

    #all loss
    trainer.extend(ex.PlotReport(['gen/loss', 'dis/loss'],
        x_key='iteration', filename='loss.png', trigger=(25, 'iteration')))

    trainer.extend(ex.PlotReport(['gen/loss', 'dis/loss', 'gen/valid_loss', 'dis/valid_loss'],
        x_key='epoch', filename='loss-with-valid.png', trigger=(2, 'epoch')))

    #other
    trainer.extend(ex.PlotReport(['dis/gp'],
        x_key='iteration', filename='gp.png', trigger=(25, 'iteration')))
    trainer.extend(ex.PlotReport(['dis/gp'],
        x_key='epoch', filename='gp(details).png', trigger=(2, 'epoch')))

    trainer.extend(ex.PlotReport(['gen/perceptual_loss'],
        x_key='iteration', filename='perceptual_loss.png', trigger=(25, 'iteration')))
    trainer.extend(ex.PlotReport(['gen/perceptual_loss'],
        x_key='epoch', filename='perceptual_loss(details).png', trigger=(2, 'epoch')))

    trainer.extend(ex.PlotReport(['gen/fm_loss'],
        x_key='iteration', filename='fm_loss.png', trigger=(25, 'iteration')))
    trainer.extend(ex.PlotReport(['gen/fm_loss'],
        x_key='epoch', filename='fm_loss(details).png', trigger=(2, 'epoch')))

    #snap
    trainer.extend(ex.snapshot_object(gen, 'gen_snapshot_epoch-{.updater.epoch}.npz'),
        trigger=(opt.snap_interval_epoch, 'epoch'))
    trainer.extend(ex.snapshot_object(dis, 'dis_snapshot_epoch-{.updater.epoch}.npz'),
        trigger=(opt.snap_interval_epoch, 'epoch'))

    trainer.extend(lambda *args: updater.save_img(),
        trigger=(opt.img_interval_iteration, 'iteration'), priority=PRIORITY_READER)

    trainer.extend(lambda *args: adam_lr_poly(opt, trainer), trigger=(100, 'iteration'))

    print('[Msg] applied extention')

    print('[Msg] start training...')
    trainer.run() #start learning


if __name__ == '__main__':
    opt = get_options()

    train(opt)

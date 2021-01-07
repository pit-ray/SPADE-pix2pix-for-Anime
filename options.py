#coding: utf-8
from argparse import ArgumentParser

def get_options():
    parser = ArgumentParser()
    parser.add_argument('--use_cpu', action='store_true', help='for debug')

    #dataset info
    parser.add_argument('--use_rough', action='store_true', help='use generated datasets by Anime-Semantic-Segmentation-GAN.')
    parser.add_argument('--dataset_dir', type=str, default='datasets/full_hc', help='directory of dataset for normal training')
    parser.add_argument('--rough_dataset_dir', type=str, default='datasets/resnet-large_hc', help='directory of generated dataset for normal training')
    parser.add_argument('--augment_data', type=bool, default=True, help='if this flag is true, annotated data is augmented.')
    parser.add_argument('--img_shape', type=tuple, default=(3, 256, 256), help='generator output image shape. this tuple is limited int elements. the order is (channels, heights, widths)')
    parser.add_argument('--c_shape', type=tuple, default=(3, 256, 256), help='conditonal image shape. this tuple is limited int elements. the order is (channels, heights, widths)')
    parser.add_argument('--class_num', type=int, default=4, help='target object class of semantic segmentation without background.')

    #training hyper-parameter
    parser.add_argument('--batch_size', type=int, default=1, help='the number of traing samples utilized in one iteration.')

    parser.add_argument('--g_lr', type=float, default=0.5*1e-4, help='learning rate of adam optimizer in order to train generator')
    parser.add_argument('--g_beta1', type=float, default=0.0, help='beta1 of adam optimizer in order to train generator')
    parser.add_argument('--g_beta2', type=float, default=0.9, help='beta2 of adam optimizer in order to train generator')

    parser.add_argument('--d_lr', type=float, default=2.0*1e-4, help='learning rate of adam optimizer in order to train discriminator')
    parser.add_argument('--d_beta1', type=float, default=0.0, help='beta1 of adam optimizer in order to train discriminator')
    parser.add_argument('--d_beta2', type=float, default=0.9, help='beta2 of adam optimizer in order to train discriminator')

    parser.add_argument('--max_iteration', type=int, default=1e5, help='maximum roop number of training.')
    parser.add_argument('--snap_interval_epoch', type=int, default=50, help='interval saving model parameter')
    parser.add_argument('--img_interval_iteration', type=int, default=50, help='interval saving image sample')
    parser.add_argument('--lr_poly_train_period', type=float, default=0.4, help='if current iteration is larger than this, start learning rate poly.')
    parser.add_argument('--lr_poly_power', type=float, default=0.9, help='strongth of learning rate poly')
    parser.add_argument('--out_dir', type=str, default='result', help='directory of outputs')

    parser.add_argument('--n_dpg', type=int, default=2, help='number of training discriminator per generator.')

    #model archtecture hyper-parameter
    parser.add_argument('--conv_norm', type=str, default='spectral_norm_hook',
        help='convolution weight normalization type. [original] is typical convolution. [spectral_norm] is only used chainer.funcstions. [spectral_norm_hook] is based on chainer.link_hooks. there is details in spectral_norms.py.')
    parser.add_argument('--ngf', type=int, default=64, help='dimension of hidden feature map at generator')
    parser.add_argument('--ndf', type=int, default=64, help='dimension of hidden feature map at discriminator')
    parser.add_argument('--division_ch', type=int, default=8, help='divide channels to reduce the computational cost at SelfAttention Layer.')
    parser.add_argument('--spade_ch', type=int, default=32, help='dimenstion of hidden feature map at SPADE Layer.')

    parser.add_argument('--upsampling_mode', type=str, default='bilinear', help='bilinear|nearest|deconv|subpx_conv')
    parser.add_argument('--aspp_nf', type=int, default=256, help='dimension of hidden feature map at ASPP archtecture')

    parser.add_argument('--perceptual_model', type=str, default='VGG16', help="VGG16|VGG19")

    #loss hyper parameter
    parser.add_argument('--adv_loss_mode', type=str, default='hinge', help='adversarial loss approch. [bce] is binary-cross-entrtopy or softplus-loss. [mse] is mean-squered-error. [hinge] is hinge-loss')
    parser.add_argument('--perceptual_coef', type=float, default=10, help='perceptual loss coef')
    parser.add_argument('--perceptual_mode', type=str, default='MAE', help='MSE|MAE')
    parser.add_argument('--zero_gp_mode', type=str, default='real', help='real|fake|real_fake')
    parser.add_argument('--gp_coef', type=float, default=10, help='gradient penalty coef')
    parser.add_argument('--fm_coef', type=float, default=10, help='feature matching loss coef. (original is 10)')
    parser.add_argument('--rough_loss_weight', type=float, default=0.1, help='rough dataset is incompleter than self-anotation. Thus, its gradient must be a little smaller.')

    #for loading trained models
    parser.add_argument('--gen_snapshot', type=str, default='', help='trained model path of generator.')
    parser.add_argument('--dis_snapshot', type=str, default='', help='trained model path of discriminator.')

    return parser.parse_args()
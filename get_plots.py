import random
import os
import torch

import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
# TODO: REMOVE
import imageio as iio
################
from typing import Optional
from utils.parse_args import create_arg_parser
from wrappers.our_gen_wrapper import get_gan, save_model
from data_loaders.prepare_data import create_data_loaders
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from wrappers.our_gen_wrapper import load_best_gan

best_ssims = [159, 220, 621, 518, 151, 33, 431, 835, 575, 649, 763, 522, 652, 343, 594, 711, 985, 972, 339, 374, 190, 590, 958, 580, 956]
best_lpips = [159, 133, 655, 522, 985, 267, 220, 594, 94, 707, 845, 984, 198, 112, 623, 565, 625, 41, 580, 467, 747, 992, 898, 733, 234]
best_dists = [460, 904, 468, 401, 126, 862, 984, 987, 577, 554, 97, 592, 733, 990, 605, 349, 178, 669, 647, 332, 579, 635, 985, 429, 512]

def psnr(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)

    return psnr_val


def ssim(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    # if not gt.ndim == 3:
    #   raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = structural_similarity(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), data_range=maxval, multichannel=True
    )

    return ssim

def gif_im(true, y, gen_im_ours, gen_im_comod, index, type, disc_num=False):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
    fig.suptitle('Dev Example')
    ax1.imshow(true.cpu().numpy().transpose(1, 2, 0))
    ax1.set_title('x')
    ax2.imshow(y.cpu().numpy().transpose(1, 2, 0))
    ax2.set_title('y')
    ax3.imshow(gen_im_ours.cpu().numpy().transpose(1, 2, 0))
    ax3.set_title(f'Ours Z {index}')
    ax4.imshow(gen_im_comod.cpu().numpy().transpose(1, 2, 0))
    ax4.set_title(f'CoModGAN Z {index}')

    plt.savefig(f'gif_{type}_{index - 1}.png')
    plt.close(fig)


def generate_gif(args, type, ind, num):
    images = []
    for i in range(num):
        images.append(iio.imread(f'gif_{type}_{i}.png'))

    iio.mimsave(f'test_ims/variation_gif_{ind}.gif', images, duration=0.25)

    for i in range(num):
        os.remove(f'gif_{type}_{i}.png')

def get_plots(args, G_ours, G_comod, test_loader, truncation, truncation_latent):
    total = 0
    fig_count = 0
    num_code = 32
    for i, data in enumerate(test_loader):
        G_ours.update_gen_status(val=True)
        G_comod.update_gen_status(val=True)
        with torch.no_grad():
            x, y, mean, std, mask = data[0]

            mean = mean.cuda()
            std = std.cuda()
            mask = mask.cuda()
            y = y.to(args.device)
            x = x.to(args.device)

            gens_ours = torch.zeros(size=(y.size(0), num_code, args.in_chans, args.im_size, args.im_size),
                               device=args.device)
            gens_comod_psi_1 = torch.zeros(size=(y.size(0), num_code, args.in_chans, args.im_size, args.im_size),
                                    device=args.device)
            for z in range(num_code):
                gens_ours[:, z, :, :, :] = G_ours(y, x=x, mask=mask, truncation=None, truncation_latent=None) * std[:, :, None, None] + mean[:, :, None, None]
                gens_comod_psi_1[:, z, :, :, :] = G_comod(y, x=x, mask=mask, truncation=None, truncation_latent=None) * std[:, :, None, None] + mean[:, :, None, None]

            # avg_ours = torch.mean(gens_ours, dim=1)
            # avg_comod_psi_1 = torch.mean(gens_comod_psi_1, dim=1)
            x = x * std[:, :, None, None] + mean[:, :, None, None]
            y_unnorm = y * std[:, :, None, None] + mean[:, :, None, None]

            for j in range(y.size(0)):
                total += 1

                if total in best_lpips:
                    fig_count += 1

                    fig = plt.figure()
                    plt.axis('off')
                    plt.imshow(x[j, :, :, :].cpu().numpy().transpose(1, 2, 0))
                    plt.savefig(f'neurips_plots/lpips/original_{fig_count}.png', bbox_inches='tight', dpi=300)
                    plt.close(fig)

                    fig = plt.figure()
                    plt.axis('off')
                    plt.imshow(y_unnorm[j, :, :, :].cpu().numpy().transpose(1, 2, 0))
                    plt.savefig(f'neurips_plots/lpips/masked_{fig_count}.png', bbox_inches='tight', dpi=300)
                    plt.close(fig)

                    # fig, ax1 = plt.subplots(1, 1)
                    # ax1.set_xticks([])
                    # ax1.set_yticks([])
                    # ax2.set_xticks([])
                    # ax2.set_yticks([])
                    # ax3.set_xticks([])
                    # ax3.set_yticks([])
                    # ax4.set_xticks([])
                    # ax4.set_yticks([])
                    # fig.suptitle(f'Test Example {fig_count}')
                    # ax1.imshow(x[j, :, :, :].cpu().numpy().transpose(1, 2, 0))
                    # ax1.set_xlabel('Original', fontweight='bold')
                    # ax2.imshow(y_unnorm[j, :, :, :].cpu().numpy().transpose(1, 2, 0))
                    # ax2.set_xlabel('Masked', fontweight='bold')
                    # ax3.imshow(avg_ours[j, :, :, :].cpu().numpy().transpose(1, 2, 0))
                    # ax3.set_title('Ours')
                    # ax4.imshow(avg_comod_psi_1[j, :, :, :].cpu().numpy().transpose(1, 2, 0))
                    # ax4.set_title('CoModGAN')


                    place = 1

                    fig = plt.figure()
                    fig.subplots_adjust(wspace=0, hspace=0.05)

                    for r in range(5):
                        ax = fig.add_subplot(1, 5, r+1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # if r == 2:
                        #     ax.set_xlabel('Ours',fontweight='bold')
                        ax.imshow(gens_ours[j, r, :, :, :].cpu().numpy().transpose(1, 2, 0))

                    plt.savefig(f'neurips_plots/lpips/5_recons_ours_{fig_count}.png',bbox_inches='tight', dpi=300)
                    plt.close(fig)

                    fig = plt.figure()
                    fig.subplots_adjust(wspace=0, hspace=0.05)

                    for r in range(5):
                        ax = fig.add_subplot(1, 5, r+1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # if r == 2:
                        #     ax.set_xlabel('CoModGAN',fontweight='bold')
                        ax.imshow(gens_comod_psi_1[j, r, :, :, :].cpu().numpy().transpose(1, 2, 0))

                    plt.savefig(f'neurips_plots/lpips/5_recons_comodgan_{fig_count}.png',bbox_inches='tight', dpi=300)
                    plt.close(fig)


                    # for r in range(num_code):
                    #     gif_im(x[j, :, :, :], y_unnorm[j, :, :, :],
                    #            gens_ours[j, r, :, :, :], gens_comod_psi_1[j, r, :, :, :], place,
                    #            'image')
                    #     place += 1
                    #
                    # generate_gif(args, 'image', fig_count, num_code)


if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    torch.backends.cudnn.benchmark = True

    args = create_arg_parser().parse_args()
    # restrict visible cuda devices
    if args.data_parallel or (args.device >= 0):
        if not args.data_parallel:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    args.in_chans = 3
    args.out_chans = 3

    args.checkpoint_dir = '/home/bendel.8/Git_Repos/celeb-inpaint/trained_models/cvpr_ours_256'
    G_ours = load_best_gan(args)

    args.checkpoint_dir = '/home/bendel.8/Git_Repos/celeb-inpaint/trained_models/cvpr_comodgan_256'
    G_comod = load_best_gan(args)

    _, _, test_loader = create_data_loaders(args)

    truncation_latent = None

    for i, data in enumerate(test_loader):
        x, y, mean, std, mask = data[0]
        x = x.cuda()
        y = y.cuda()
        mask = mask.cuda()
        mean = mean.cuda()
        std = std.cuda()

        truncation_latent = torch.mean(G_ours.get_mean_code_vector(y, x, mask, num_latents=128), dim=0)
        break

    get_plots(args, G_ours, G_comod, test_loader, None, None)

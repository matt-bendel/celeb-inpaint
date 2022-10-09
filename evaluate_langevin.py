import random
import os
import torch
import time

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
from evaluation_scripts.cfid.embeddings import InceptionEmbedding
from evaluation_scripts.cfid.cfid_metric_langevin import CFIDMetric
from evaluation_scripts.fid.fid_metric_langevin import FIDMetric

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

def gif_im(true, gen_im, index, type, disc_num=False):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.suptitle('Dev Example')
    ax1.imshow(true.cpu().numpy().transpose(1, 2, 0))
    ax1.set_title('GT')
    ax2.imshow(gen_im.cpu().numpy().transpose(1, 2, 0))
    ax2.set_title(f'Z {index}')

    plt.savefig(f'gif_{type}_{index - 1}.png')
    plt.close(fig)


def generate_gif(args, type, ind, num):
    images = []
    for i in range(num):
        images.append(iio.imread(f'gif_{type}_{i}.png'))

    iio.mimsave(f'test_ims/variation_gif_{ind}.gif', images, duration=0.25)

    for i in range(num):
        os.remove(f'gif_{type}_{i}.png')

def get_metrics(args, num_code):
    losses = {
        'psnr': [],
        'ssim': [],
        'apsd': [],
        '1-psnr': [],
    }
    means = {
        'psnr': [],
        'ssim': []
    }

    total = 0
    fig_count = 0
    for i in range(1000):
        total += 1

        gens = torch.zeros(size=(num_code, 3, 128, 128))
        recon_object = None

        for z in range(num_code):
            recon_object = torch.load(f'/storage/celebA-HQ/langevin_recons/image_{i}_sample_{z}.pt')
            gens[z, :, :, :] = recon_object['x_hat']

        gt = recon_object['gt'].numpy()
        avg = torch.mean(gens, dim=0)
        apsd = torch.std(gens, dim=0).mean().cpu().numpy()
        losses['apsd'].append(apsd)
        losses['ssim'].append(ssim(gt, avg.cpu().numpy()))
        losses['psnr'].append(psnr(gt, avg.cpu().numpy()))
        losses['psnr_1'].append(psnr(gt, gens[0].cpu().numpy()))

        if total % 25 == 0:
            fig_count += 1

            fig = plt.figure()
            fig.subplots_adjust(wspace=0, hspace=0.05)

            for r in range(5):
                ax = fig.add_subplot(5, 1, r + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                # if r == 2:
                #     ax.set_xlabel('Ours',fontweight='bold')
                ax.imshow(gens[r, :, :, :].cpu().numpy().transpose(1, 2, 0))

            plt.savefig(f'test_ims_2/5_recons_langevin_{fig_count}', bbox_inches='tight', dpi=300)
            plt.close(fig)

            fig = plt.figure()
            plt.imshow(gt.transpose(1, 2, 0))
            plt.savefig(f'langevin_gt_{fig_count}.png')
            plt.close(fig)

        if total % 50 == 0:
            # continue
            means['psnr'].append(np.mean(losses['psnr']))
            means['ssim'].append(np.mean(losses['ssim']))
            losses['psnr'] = []
            losses['ssim'] = []



    print(f'RESULTS for {num_code} code vectors')
    print(f'SSIM: {np.mean(means["ssim"])} \\pm {np.std(means["ssim"]) / np.sqrt(len(means["ssim"]))}')
    print(f'PSNR: {np.mean(means["psnr"])} \\pm {np.std(means["psnr"]) / np.sqrt(len(means["ssim"]))}')
    print(f'1-PSNR: {np.mean(losses["1-psnr"])}')
    print(f'APSD: {np.mean(losses["apsd"])}')


def get_cfid(args, num_samps):
    print("GETTING INCEPTION EMBEDDING")
    inception_embedding = InceptionEmbedding(parallel=True)

    print("GETTING DATA LOADERS")
    cfid_metric = CFIDMetric(gan=None,
                             loader=None,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=args,
                             num_samps=num_samps)

    print(f'{num_samps}-CFID')
    print('CFID: ', cfid_metric.get_cfid_torch())

def get_fid(args, train_loader):
    print("GETTING INCEPTION EMBEDDING")
    inception_embedding = InceptionEmbedding(parallel=True)

    print("GETTING DATA LOADERS")
    fid_metric = FIDMetric(gan=None,
                            ref_loader=train_loader,
                             loader=None,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=args)

    print('FID: ', fid_metric.get_fid())

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

    train_loader, _, _ = create_data_loaders(args)
    # get_cfid(args, 1)
    # get_fid(args, train_loader)
    vals = [32]
    for val in vals:
        get_metrics(args, val)

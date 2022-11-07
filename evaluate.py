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
from evaluation_scripts.cfid.cfid_metric import CFIDMetric
from evaluation_scripts.fid.fid_metric import FIDMetric
from evaluation_scripts.lpips.lpips_metric import LPIPSMetric

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

def get_metrics(args, G, test_loader, num_code, truncation=None):
    losses = {
        'psnr': [],
        'ssim': [],
        'apsd': []
    }
    means = {
        'psnr': [],
        'ssim': []
    }
    times = []

    total = 0
    fig_count = 0
    for i, data in enumerate(test_loader):
        G.update_gen_status(val=True)
        with torch.no_grad():
            x, y, mean, std, mask = data[0]

            mean = mean.cuda()
            std = std.cuda()
            mask = mask.cuda()
            y = y.to(args.device)
            x = x.to(args.device)

            gens = torch.zeros(size=(y.size(0), num_code, args.in_chans, args.im_size, args.im_size),
                               device=args.device)
            for z in range(num_code):
                start = time.time()
                gens[:, z, :, :, :] = G(y, x=x, mask=mask, truncation=None, truncation_latent=None)  * std[:, :, None, None] + mean[:, :, None, None]
                elapsed = time.time() - start
                times.append(elapsed)

            avg = torch.mean(gens, dim=1)
            x = x * std[:, :, None, None] + mean[:, :, None, None]
            y_unnorm = y * std[:, :, None, None] + mean[:, :, None, None]

            for j in range(y.size(0)):
                total += 1
                losses['ssim'].append(ssim(x[j].cpu().numpy(), avg[j].cpu().numpy()))
                losses['psnr'].append(psnr(x[j].cpu().numpy(), avg[j].cpu().numpy()))
                y_gens = gens[j, :, :, :, :]
                std_tensor = torch.std(y_gens, dim=0)
                losses['apsd'].append(std_tensor.mean().cpu().numpy())
                if total % 50 == 0:
                    # continue
                    fig_count += 1
                    means['psnr'].append(np.mean(losses['psnr']))
                    means['ssim'].append(np.mean(losses['ssim']))
                    losses['psnr'] = []
                    losses['ssim'] = []


    print(f'RESULTS for {num_code} code vectors')
    print(f'SSIM: {np.mean(means["ssim"])} \\pm {np.std(means["ssim"]) / np.sqrt(len(means["ssim"]))}')
    print(f'PSNR: {np.mean(means["psnr"])} \\pm {np.std(means["psnr"]) / np.sqrt(len(means["ssim"]))}')
    print(f'APSD: {np.mean(losses["apsd"])}')
    print(f'TIME: {np.mean(times)}')


def get_cfid(args, G, test_loader, num_samps, dev_loader, train_loader):
    print("GETTING INCEPTION EMBEDDING")
    inception_embedding = InceptionEmbedding(parallel=True)

    print("GETTING DATA LOADERS")
    cfid_metric = CFIDMetric(gan=G,
                             loader=test_loader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=args,
                             num_samps=num_samps,
                             truncation=None,
                             truncation_latent=None,
                             dev_loader=dev_loader,
                             train_loader=train_loader)

    cfid = cfid_metric.get_cfid_torch_pinv()
    print('CFID: ', cfid)

    return cfid

def get_fid(args, G, test_loader, train_loader, t, truncation_latent=None):
    print("GETTING INCEPTION EMBEDDING")
    inception_embedding = InceptionEmbedding(parallel=True)

    print("GETTING DATA LOADERS")
    fid_metric = FIDMetric(gan=G,
                            ref_loader=train_loader,
                             loader=test_loader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=args,
                             truncation=t,
                             truncation_latent=truncation_latent)

    fid = fid_metric.get_fid()
    print('FID: ', fid)
    return fid

def get_lpips(args, G, test_loader, num_runs, t, truncation_latent=None):
    lpips_metric = LPIPSMetric(G, test_loader)
    LPIPS = lpips_metric.compute_lpips(num_runs, t, truncation_latent)
    print('LPIPS: ', LPIPS)
    return LPIPS

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

    G = load_best_gan(args)
    G.update_gen_status(val=True)

    train_loader, val_loader, test_loader = create_data_loaders(args)

    truncation_latent = None

    for i, data in enumerate(test_loader):
        x, y, mean, std, mask = data[0]
        x = x.cuda()
        y = y.cuda()
        mask = mask.cuda()
        mean = mean.cuda()
        std = std.cuda()

        truncation_latent = torch.mean(G.get_mean_code_vector(y, x, mask, num_latents=128), dim=0)
        break

    # cfid_old_svd = get_cfid(args, G, test_loader, 1, None, truncation_latent=truncation_latent, cfid_comp=0)
    # cfid_new_svd = get_cfid(args, G, test_loader, 1, None, truncation_latent=truncation_latent, cfid_comp=1)
    # cfid_pinv = get_cfid(args, G, test_loader, 1, None, truncation_latent=truncation_latent, cfid_comp=2)
    # cfid_def_pinv = get_cfid(args, G, test_loader, 1, None, truncation_latent=truncation_latent, cfid_comp=3)

    # print(f'CFID OLD SVD: {cfid_old_svd}')
    # print(f'CFID NEW SVD: {cfid_new_svd}')
    # print(f'CFID PINV: {cfid_pinv}')

    # vals = [1, 2, 4, 8, 16, 32]
    # get_fid(args, G, test_loader, train_loader, t=None, truncation_latent=None)
    # get_cfid(args, G, test_loader, 32, None, None)
    # get_cfid(args, G, test_loader, 8, val_loader, None)
    # get_cfid(args, G, test_loader, 1, val_loader, train_loader)
    # exit()
    vals = [1]
    for val in vals:
        get_metrics(args, G, test_loader, val, truncation=None)

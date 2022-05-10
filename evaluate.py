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
from evaluation_scripts.cfid.embeddings import InceptionEmbedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric

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


def get_metrics(args, G, test_loader):
    losses = {
        'psnr': [],
        'ssim': []
    }
    means = {
        'psnr': [],
        'ssim': []
    }

    total = 0
    for i, data in enumerate(test_loader):
        G.update_gen_status(val=True)
        with torch.no_grad():
            x, y, mean, std = data[0]
            mean = mean.cuda()
            std = std.cuda()
            y = y.to(args.device)
            x = x.to(args.device)

            gens = torch.zeros(size=(y.size(0), args.num_z, args.in_chans, args.im_size, args.im_size),
                               device=args.device)
            for z in range(128):
                gens[:, z, :, :, :] = G(y)

            avg = torch.mean(gens, dim=1) * std[:, :, None, None] + mean[:, :, None, None]
            x = x * std[:, :, None, None] + mean[:, :, None, None]

            for j in range(y.size(0)):
                total += 1
                losses['ssim'].append(ssim(x[j].cpu().numpy(), avg[j].cpu().numpy()))
                losses['psnr'].append(psnr(x[j].cpu().numpy(), avg[j].cpu().numpy()))
                if total % 100 == 0:
                    means['psnr'].append(np.mean(losses['psnr']))
                    means['ssim'].append(np.mean(losses['ssim']))
                    losses['psnr'] = []
                    losses['ssim'] = []

    print('RESULTS')
    print(f'SSIM: {np.mean(means["ssim"])} \\pm {np.std(means["ssim"])}')
    print(f'PSNR: {np.mean(means["psnr"])} \\pm {np.std(means["psnr"])}')


def get_cfid(args, G, test_loader):
    print("GETTING INCEPTION EMBEDDING")
    inception_embedding = InceptionEmbedding(parallel=True)

    print("GETTING DATA LOADERS")
    cfid_metric = CFIDMetric(gan=G,
                             loader=test_loader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=args)

    print('CFID: ', cfid_metric.get_cfid_torch())


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

    if not args.MSE and not args.L1:
        args.model_num = 1
    elif args.MSE and not args.VAR:
        args.model_num = 2
    elif args.MSE and args.VAR:
        args.model_num = 3
    else:
        args.model_num = 4

    print(f'MODEL NUM: {args.model_num}')

    args.in_chans = 3
    args.out_chans = 3

    G = load_best_gan(args)

    _, _, test_loader = create_data_loaders(args)
    get_metrics(args, G, test_loader)
    get_cfid(args, G, test_loader)

import random
import os
import torch
import time
import itertools

import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
# TODO: REMOVE
import imageio as iio
################
from typing import Optional
import sklearn.preprocessing
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
from DISTS_pytorch import DISTS

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

    args.checkpoint_dir = '/storage/matt_models/inpainting/cvpr_ours_256'

    G = load_best_gan(args)
    G.update_gen_status(val=True)

    train_loader, val_loader, test_loader = create_data_loaders(args)
    num_code = 64
    current_count = 0
    count = 5

    G.update_gen_status(val=True)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, y, mean, std, mask = data[0]

            mean = mean.cuda()
            std = std.cuda()
            mask = mask.cuda()
            y = y.cuda()
            x = x.cuda()

            gens = torch.zeros(size=(y.size(0), num_code, args.in_chans, 256, 256),
                               device=args.device)
            for z in range(num_code):
                gens[:, z, :, :, :] = G(y, x=x, mask=mask, truncation=None, truncation_latent=None) * std[:, :, None,
                                                                                                      None] + mean[:, :,
                                                                                                              None,
                                                                                                              None]
            for j in range(y.shape[0]):
                if current_count >= count:
                    exit()

                single_samps = np.zeros((num_code, 3, 256, 256))
                np_avg = torch.mean(gens[j, :, :, :, :], dim=0).cpu().numpy()
                for z in range(num_code):
                    single_samps[z, :, :, :] = gens[j, z, :, :, :].cpu().numpy()

                single_samps = single_samps - np_avg[None, :, :, :]

                cov_mat = np.zeros((num_code, 3*np_avg.shape[-1] * np_avg.shape[-2]))

                for z in range(num_code):
                    cov_mat[z, :] = single_samps[z].flatten()

                u, s, vh = np.linalg.svd(cov_mat, full_matrices=False)

                plt.figure()
                plt.scatter(range(len(s)), sklearn.preprocessing.normalize(s.reshape((1, -1))))
                plt.savefig(f'sv_test/test_sv_{current_count}.png')
                plt.close()

                for l in range(2):
                    v_re = vh[l].reshape((3, 256, 256))
                    v_re = (v_re - np.min(v_re)) / (np.max(v_re) - np.min(v_re))
                    plt.figure()
                    plt.imshow(v_re.transpose(1, 2, 0))
                    plt.savefig(f'sv_test/test_sv_v_{num_code}_{current_count}_{l}.png')
                    plt.close()

                exit()
                current_count += 1
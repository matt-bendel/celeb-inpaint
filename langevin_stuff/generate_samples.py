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

@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size, mask,
                                        n_steps_each=100, step_lr=0.000008):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """

    images = []

    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().view(-1, 3, image_size, image_size)
    x_mod = x_mod.view(-1, 3, image_size, image_size)
    masked_image = refer_image * mask
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                corrupted_half_image = masked_image + torch.randn_like(masked_image) * sigma
                x_mod[:, :, :, :cols] = corrupted_half_image
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                                                                         grad.abs().max()))

        return images

def sample(self):
    if self.config.sampling.ckpt_id is None:
        states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
    else:
        states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                            map_location=self.config.device)

    score = get_model(self.config)
    score = torch.nn.DataParallel(score)

    score.load_state_dict(states[0], strict=True)

    if self.config.model.ema:
        ema_helper = EMAHelper(mu=self.config.model.ema_rate)
        ema_helper.register(score)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(score)

    sigmas_th = get_sigmas(self.config)
    sigmas = sigmas_th.cpu().numpy()

    transform = transforms.Compose([transforms.ToTensor(), DataTransform(args)])
    dataset = datasets.ImageFolder('/storage/celebA-HQ/celeba_hq_128', transform=transform)
    _, _, test_data = torch.utils.data.random_split(
        dataset, [27000, 2000, 1000],
        generator=torch.Generator().manual_seed(0)
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    score.eval()

    num_samps = 32
    for i, data in enumerate(test_loader):
        x, y, mean, std, mask = data[0]
        x = x.to(self.config.device)

        for j in range(num_samps):
            width = int(np.sqrt(self.config.sampling.batch_size))
            init_samples = torch.rand(width, width, self.config.data.channels,
                                      self.config.data.image_size,
                                      self.config.data.image_size,
                                      device=self.config.device)
            init_samples = data_transform(self.config, init_samples)
            all_samples = anneal_Langevin_dynamics_inpainting(init_samples, refer_images[:width, ...], score,
                                                              sigmas,
                                                              self.config.data.image_size,
                                                              self.config.sampling.n_steps_each,
                                                              self.config.sampling.step_lr)

            torch.save(refer_images[:width, ...], os.path.join(self.args.image_folder, 'refer_image.pth'))
            refer_images = refer_images[:width, None, ...].expand(-1, width, -1, -1, -1).reshape(-1,
                                                                                                 *refer_images.shape[
                                                                                                  1:])
            save_image(refer_images, os.path.join(self.args.image_folder, 'refer_image.png'), nrow=width)

            sample = all_samples[-1].view(self.config.sampling.batch_size, self.config.data.channels,
                                          self.config.data.image_size,
                                          self.config.data.image_size)

            sample = inverse_data_transform(self.config, sample)

            image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
            save_image(image_grid, os.path.join(self.args.image_folder,
                                                'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
            torch.save(sample, os.path.join(self.args.image_folder,
                                            'completion_{}.pth'.format(self.config.sampling.ckpt_id)))


_, _, test_loader = create_data_loaders(args)



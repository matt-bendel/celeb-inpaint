import os
import numpy as np
import torch
from evaluation_scripts.lpips.dist_model import DistModel
from tqdm import tqdm

class LPIPSMetric:
    def __init__(self):
        self.model = PerceptualLoss(model='net-lin',net='alex')

    def compute_lpips(self, num_runs):
        meta_dists = []
        for i in range(num_runs):
            dists = []
            for j in range(8):
                if j ==  7:
                    embedImg1 = torch.zeros(104, 3, 128, 128).cuda()
                    embedImg2 = torch.zeros(104, 3, 128, 128).cuda()
                else:
                    embedImg1 = torch.zeros(128, 3, 128, 128).cuda()
                    embedImg2 = torch.zeros(128, 3, 128, 128).cuda()

                batch_size = embedImg1.shape[0]
                for k in range(batch_size):
                    recon_object = torch.load(f'/storage/celebA-HQ/langevin_recons/image_{j*128 + k}_sample_{i+0}.pt')
                    im1 = recon_object['x_hat'].cuda()
                    embedImg1[k, :, :, :] = 2 * (im1 - torch.min(im1)) / (torch.max(im1) - torch.min(im1)) - 1

                    recon_object = torch.load(f'/storage/celebA-HQ/langevin_recons/image_{j*128 + k}_sample_{i+2}.pt')
                    im2 = recon_object['x_hat'].cuda()
                    embedImg2[k, :, :, :] = 2 * (im2 - torch.min(im2)) / (torch.max(im2) - torch.min(im2)) - 1

                dists.append(np.mean(self.model.forward(embedImg1.to("cuda:0"), embedImg2.to("cuda:0")).data.cpu().squeeze().numpy()))
                print(dists)

            meta_dists.append(np.mean(dists))

        return np.mean(meta_dists)


class PerceptualLoss(torch.nn.Module):
    def __init__(self, model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0]): # VGG using our perceptually-learned weights (LPIPS metric)
    # def __init__(self, model='net', net='vgg', use_gpu=True): # "default" way of using VGG as a perceptual loss
        super(PerceptualLoss, self).__init__()
        print('Setting up Perceptual loss...')
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu, colorspace=colorspace, spatial=self.spatial, gpu_ids=gpu_ids)
        print('...[%s] initialized'%self.model.name())
        print('...Done')

    def forward(self, pred, target):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]
        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """

        return self.model.forward(target, pred)
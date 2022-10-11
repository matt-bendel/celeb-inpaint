import os
import numpy as np
import torch
from evaluation_scripts.lpips.dist_model import DistModel
from tqdm import tqdm

class LPIPSMetric:
    def __init__(self, G, data_loader):
        self.G = G
        self.G.update_gen_status(val=True)
        self.loader = data_loader
        self.model = PerceptualLoss(model='net-lin',net='alex')

    def compute_lpips(self, num_runs):
        meta_dists = []
        for i in range(num_runs):
            dists = []
            for j, data in tqdm(enumerate(self.loader),
                                desc='Computing generated distribution',
                                total=len(self.loader)):
                x, y, mean, std, mask = data[0]
                x = x.cuda()
                y = y.cuda()
                mask = mask.cuda()
                mean = mean.cuda()
                std = std.cuda()

                img1 = self.G(y, x=x, mask=mask, truncation=None)
                img2 = self.G(y, x=x, mask=mask, truncation=None)

                embedImg1 = torch.zeros(size=(img1.size(0), 3, 128, 128)).cuda()
                embedImg2 = torch.zeros(size=(img2.size(0), 3, 128, 128)).cuda()

                for l in range(img1.size(0)):
                    im1 = img1[l, :, :, :] * std[l, :, None, None] + mean[l, :, None, None]
                    im1 = 2 * (im1 - torch.min(im1)) / (torch.max(im1) - torch.min(im1)) - 1
                    embedImg1[l, :, :, :] = im1

                    im2 = img2[l, :, :, :] * std[l, :, None, None] + mean[l, :, None, None]
                    im2 = 2 * (im2 - torch.min(im2)) / (torch.max(im2) - torch.min(im2)) - 1
                    embedImg2[l, :, :, :] = im2

                dists.append(np.mean(self.model.forward(embedImg1.to("cuda:0"), embedImg2.to("cuda:0")).data.cpu().squeeze().numpy()))

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
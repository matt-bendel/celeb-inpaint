import os
import numpy as np
import torch
from evaluation_scripts.lpips.dist_model import DistModel
from tqdm import tqdm
import matplotlib.pyplot as plt

class LPIPSMetric:
    def __init__(self, G, data_loader):
        self.G = G
        self.G.update_gen_status(val=True)
        self.loader = data_loader
        self.model = PerceptualLoss(model='net-lin',net='alex')

    def compute_lpips(self, num_runs, truncation, truncation_latent):
        meta_dists = []
        im_dict = {}

        for i in range(1):
            break
            dists = []
            total = 0
            for j, data in tqdm(enumerate(self.loader),
                                desc='Computing generated distribution',
                                total=len(self.loader)):
                x, y, mean, std, mask = data[0]
                x = x.cuda()
                y = y.cuda()
                mask = mask.cuda()
                mean = mean.cuda()
                std = std.cuda()
                samp_count = 32
                lpips_vals = np.zeros((y.size(0), samp_count))
                lpips_vals_lang = np.zeros((y.size(0), samp_count))

                langevin_ims = torch.zeros(size=(x.size(0), samp_count, 3, 256, 256)).cuda()
                langevin_x = torch.zeros(size=(x.size(0), 3, 256, 256)).cuda()

                for k in range(samp_count):
                    img1 = self.G(y, x=x, mask=mask, truncation=None, truncation_latent=None)

                    embedImg1 = torch.zeros(size=(img1.size(0), 3, 256, 256)).cuda()
                    embedImg2 = torch.zeros(size=(img1.size(0), 3, 256, 256)).cuda()

                    embedImgLang1 = torch.zeros(size=(img1.size(0), 3, 256, 256)).cuda()
                    embedImgLang2 = torch.zeros(size=(img1.size(0), 3, 256, 256)).cuda()

                    for l in range(img1.size(0)):
                        recon_object = torch.load(f'/storage/celebA-HQ/langevin_recons_256/image_{j*20 + l}_sample_{k}.pt')
                        langevin_ims[l, k, :, :, :] = recon_object['x_hat']
                        if k == 0:
                            langevin_x[l, :, :, :] = recon_object['gt']

                        im1 = img1[l, :, :, :] * std[l, :, None, None] + mean[l, :, None, None]
                        im1 = 2 * (im1 - torch.min(im1)) / (torch.max(im1) - torch.min(im1)) - 1
                        embedImg1[l, :, :, :] = im1

                        im2 = x[l, :, :, :] * std[l, :, None, None] + mean[l, :, None, None]
                        im2 = 2 * (im2 - torch.min(im2)) / (torch.max(im2) - torch.min(im2)) - 1
                        embedImg2[l, :, :, :] = im2

                        embedImgLang1[l, :, :, :] = 2*(langevin_x[l, :, :, :] - torch.min(langevin_x[l, :, :, :])) / (torch.max(langevin_x[l, :, :, :]) - torch.min(langevin_x[l, :, :, :])) - 1
                        embedImgLang2[l, :, :, :] = 2*(langevin_ims[l, k, :, :, :] - torch.min(langevin_ims[l, k, :, :, :])) / (torch.max(langevin_ims[l, k, :, :, :]) - torch.min(langevin_ims[l, k, :, :, :])) - 1

                    lpips_vals[:, k] = self.model.forward(embedImg1.to("cuda:0"), embedImg2.to("cuda:0")).data.cpu().squeeze().numpy()
                    lpips_vals_lang[:, k] = self.model.forward(embedImgLang1.to("cuda:0"), embedImgLang2.to("cuda:0")).data.cpu().squeeze().numpy()

                for l in range(lpips_vals.shape[0]):
                    total += 1
                    im_dict[total] = np.max(lpips_vals[l, :])#np.min(lpips_vals[l, :] - lpips_vals_lang)#

        sorted_dict = dict(sorted(im_dict.items(), key=lambda x: x[1], reverse=True)[-25:])
        print(sorted_dict.keys())
        # TODO: CONVERT TO DICT
        total = 0
        fig_count = 0
        with torch.no_grad():
            for j, data in tqdm(enumerate(self.loader),
                                desc='Computing generated distribution',
                                total=len(self.loader)):
                x, y, mean, std, mask = data[0]
                x = x.cuda()
                y = y.cuda()
                mask = mask.cuda()
                mean = mean.cuda()
                std = std.cuda()
                samp_count = 32
                recons = torch.zeros(1, samp_count, 3, 256, 256).cuda()
                lpips_vals = np.zeros((1, samp_count))

                temp_count = total + y.size(0)

                if temp_count < 732:
                    total += y.size(0)
                    fig_count += y.size(0)
                    continue

                no_valid = False
                valid_inds = []

                for k in range(samp_count):
                    img1 = self.G(y, x=x, mask=mask, truncation=None, truncation_latent=None)

                    embedImg1 = torch.zeros(size=(img1.size(0), 3, 256, 256)).cuda()
                    embedImg2 = torch.zeros(size=(img1.size(0), 3, 256, 256)).cuda()

                    for l in range(img1.size(0)):
                        if k == 0:
                            total += 1

                            valid_inds.append(l)
                        else:
                            if l not in valid_inds:
                                continue

                        im1 = img1[l, :, :, :] * std[l, :, None, None] + mean[l, :, None, None]
                        im1 = 2 * (im1 - torch.min(im1)) / (torch.max(im1) - torch.min(im1)) - 1
                        embedImg1[l, :, :, :] = im1

                        im2 = x[l, :, :, :] * std[l, :, None, None] + mean[l, :, None, None]
                        im2 = 2 * (im2 - torch.min(im2)) / (torch.max(im2) - torch.min(im2)) - 1
                        embedImg2[l, :, :, :] = im2

                    if len(valid_inds) == 0:
                        no_valid = True
                        break

                    newEmbed1 = torch.zeros(len(valid_inds), 3, 256, 256).cuda()
                    newEmbed2 = torch.zeros(len(valid_inds), 3, 256, 256).cuda()
                    newRecons = torch.zeros(len(valid_inds), 3, 256, 256).cuda()
                    if k == 0:
                        lpips_vals = np.repeat(lpips_vals, len(valid_inds), axis=0)

                    new_count = 0
                    for valid_idx in valid_inds:
                        newRecons[new_count, :, :, :] = img1[valid_idx, :, :, :] * std[valid_idx, :, None, None] + mean[valid_idx, :, None, None]
                        newEmbed1[new_count, :, :, :] = embedImg1[valid_idx, :, :, :]
                        newEmbed2[new_count, :, :, :] = embedImg2[valid_idx, :, :, :]
                        new_count += 1

                    if k == 0:
                        recons = torch.zeros(1, samp_count, 3, 256, 256).cuda()
                        recons = recons.repeat(len(valid_inds), 1, 1, 1, 1)

                    recons[:, k, :, :, :] = newRecons[:, :, :, :]

                    lpips_vals[:, k] = self.model.forward(newEmbed1.to("cuda:0"), newEmbed2.to("cuda:0")).data.cpu().squeeze().numpy()

                if no_valid:
                    continue
                else:
                    print("Valid inds...")

                for l in range(lpips_vals.shape[0]):
                    fig_count += 1
                    lth_vals = np.array(lpips_vals[l, :])

                    idx = np.argpartition(lth_vals, 30)

                    if fig_count == 732:
                        fig = plt.figure()
                        fig.subplots_adjust(wspace=0, hspace=0.05)

                        tc = 1
                        samp_nums = [31, 30, 29, 26, 25, 24, 23, 21, 20, 18, 17, 16, 13 ,12 ,10, 8, 7]
                        subsamp_nums = [3, 6, 7, 8, 10, 12, 15, 16]
                        subsubsamp_nums = [0, 1, 2, 3, 7]
                        # [0, 4, 5, 6, 20, 22, 23, 24, 25, 27]
                        for r in [3,5,6,15,16]:
                            ax = fig.add_subplot(1, 5, tc)
                            tc += 1
                            ax.set_xticks([])
                            ax.set_yticks([])
                            # if r == 2:
                            #     ax.set_xlabel('Ours',fontweight='bold')
                            ax.imshow(recons[l, idx[r], :, :, :].cpu().numpy().transpose(1, 2, 0))

                        plt.savefig(f'neurips_plots/test_ours/5_recons_ours_{fig_count}.png', bbox_inches='tight', dpi=300)
                        plt.close(fig)


        sorted_dict = sorted(im_dict.items(), key=lambda x: x[1], reverse=True)[-25:]
        print(str(dict(sorted_dict[-25:])))


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
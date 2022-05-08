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


GLOBAL_LOSS_DICT = {
    'g_loss': [],
    'd_loss': [],
    'mSSIM': [],
    'd_acc': []
}


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
        gt, pred, data_range=maxval
    )

    return ssim


def compute_gradient_penalty(D, real_samples, fake_samples, args, y):
    """Calculates the gradient penalty loss for WGAN GP"""
    Tensor = torch.FloatTensor
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(args.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(input=interpolates, y=y)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0).to(args.device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# TODO: REMOVE
def gif_im(true, gen_im, index, type, disc_num=False):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Dev Example')
    ax1.imshow(gen_im.cpu().numpy().transpose(1, 2, 0))
    ax1.set_title('GT')
    ax2.imshow(true.cpu().numpy().transpose(1, 2, 0))
    ax2.set_title(f'Z {index}')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.savefig(f'/gif_{type}_{index - 1}.png')
    plt.close(fig)


def generate_gif(type):
    images = []
    for i in range(8):
        images.append(iio.imread(f'/gif_{type}_{i}.png'))

    iio.mimsave(f'variation_gif.gif', images, duration=0.25)

    for i in range(8):
        os.remove(f'/gif_{type}_{i}.png')
######################

def train(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    args.in_chans = 3
    args.out_chans = 3

    G, D, opt_G, opt_D, best_loss, start_epoch = get_gan(args)

    train_loader, dev_loader, test_loader = create_data_loaders(args)

    for epoch in range(start_epoch, args.num_epochs):
        batch_loss = {
            'g_loss': [],
            'd_loss': [],
        }

        for i, data in enumerate(train_loader):
            G.update_gen_status(val=False)
            print(data[0][0].shape)
            print(data[0][1].shape)
            print(data[0][2].shape)
            print(data[0][3].shape)
            print(data[0][4].shape)

            exit()

            x, y, mean, std, inds = data
            y = y.cuda()
            x = x.cuda()
            mean = mean.cuda()
            std = std.cuda()

            for j in range(args.num_iters_discriminator):
                for param in D.parameters():
                    param.grad = None

                x_hat = G(y, inds)

                real_pred = D(input=x, y=y)
                fake_pred = D(input=x_hat, y=y)

                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(D, x.data, x_hat.data, args, y.data)

                d_loss = torch.mean(fake_pred) - torch.mean(
                    real_pred) + args.gp_weight * gradient_penalty + 0.001 * torch.mean(real_pred ** 2)

                d_loss.backward()
                opt_D.step()

            for param in G.gen.parameters():
                param.grad = None

            gens = torch.zeros(size=(y.size(0), args.num_z, args.in_chans, args.im_size, args.im_size),
                               device=args.device)
            for z in range(args.num_z):
                gens[:, z, :, :, :] = G(y, inds)

            fake_pred = torch.zeros(size=(y.shape[0], args.num_z), device=args.device)
            for k in range(y.shape[0]):
                cond = torch.zeros(1, gens.shape[2], gens.shape[3], gens.shape[4])
                cond[0, :, :, :] = y[k, :, :, :]
                cond = cond.repeat(args.num_z, 1, 1, 1)
                temp = D(input=gens[k], y=cond)
                fake_pred[k] = temp[:, 0]

            avg_recon = torch.mean(gens, dim=1)

            gen_pred_loss = torch.mean(fake_pred[0])
            for k in range(y.shape[0] - 1):
                gen_pred_loss += torch.mean(fake_pred[k + 1])

            x = x * std[:, :, None, None] + mean[:, :, None, None]
            avg_recon = avg_recon * std[:, :, None, None] + mean[:, :, None, None]

            std_weight = np.sqrt(2 / (np.pi * args.num_z * (args.num_z + 1)))
            var_weight = 0.01
            adv_weight = 1e-4 if args.L1 else 1e-3 if args.MSE else 1
            g_loss = - adv_weight * torch.mean(gen_pred_loss)
            g_loss += F.l1_loss(avg_recon, x) if args.L1 else 0
            g_loss += - std_weight * torch.mean(torch.std(gens, dim=1), dim=(0, 1, 2, 3)) if args.L1 else 0
            g_loss += F.mse_loss(avg_recon, x) if args.MSE else 0
            g_loss += -var_weight * torch.mean(torch.var(gens, dim=1, unbiased=True), dim=(0, 1, 2, 3)) if args.VAR else 0

            g_loss.backward()
            opt_G.step()

            batch_loss['g_loss'].append(g_loss.item())
            batch_loss['d_loss'].append(d_loss.item())

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                % (epoch + 1, args.num_epochs, i, len(train_loader.dataset) / args.batch_size, d_loss.item(),
                   g_loss.item())
            )

        losses = {
            'psnr': [],
            'ssim': []
        }

        for i, data in enumerate(dev_loader):
            G.update_gen_status(val=True)
            with torch.no_grad():
                x, y, mean, std, inds = data
                mean = mean.cuda()
                std = std.cuda()
                y = y.to(args.device)
                x = x.to(args.device) * std + mean

                gens = torch.zeros(size=(y.size(0), args.num_z, args.in_chans, args.im_size, args.im_size),
                                   device=args.device)
                for z in range(args.num_z):
                    gens[:, z, :, :, :] = G(y, inds)

                avg = torch.mean(gens, dim=1) * std[:, :, None, None] + mean[:, :, None, None]
                x = x * std[:, :, None, None] + mean[:, :, None, None]

                for j in range(y.size(0)):
                    losses['ssim'].append(ssim(x.cpu().numpy(), avg.cpu().numpy()))
                    losses['psnr'].append(psnr(x.cpu().numpy(), avg.cpu().numpy()))

                if i == 0:
                    ind = 0
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.suptitle('Dev Example')
                    ax1.imshow(avg[ind, :, :, :].cpu().numpy().transpose(1, 2, 0))
                    ax1.set_title('GT')
                    ax2.imshow(x[ind, :, :, :].cpu().numpy().transpose(1, 2, 0))
                    ax2.set_title('Avg. Recon')
                    plt.savefig('dev_ex.png')
                    plt.close(fig)

                    place = 1

                    for r in range(args.num_z):
                        gif_im(x[ind, :, :, :], gens[ind, r, :, :, :] * std[0, :, None, None] + mean[0, :, None, None], place, 'image')
                        place += 1

                    generate_gif('image')
                    # TODO: ADD PLOT LOGIC - GIF AND STD. DEV

        ssim_loss = np.mean(losses['ssim'])
        best_model = ssim_loss > best_loss
        best_loss = ssim_loss if ssim_loss > best_loss else best_loss

        GLOBAL_LOSS_DICT['g_loss'].append(np.mean(batch_loss['g_loss']))
        GLOBAL_LOSS_DICT['d_loss'].append(np.mean(batch_loss['d_loss']))

        save_str = f"END OF EPOCH {epoch + 1}: [Average D loss: {GLOBAL_LOSS_DICT['d_loss'][epoch - start_epoch]:.4f}] [Average G loss: {GLOBAL_LOSS_DICT['g_loss'][epoch - start_epoch]:.4f}]\n"
        print(save_str)
        save_str_2 = f"[Avg PSNR: {np.mean(losses['psnr']):.2f}] [Avg SSIM: {np.mean(losses['ssim']):.4f}]"
        print(save_str_2)

        save_model(args, epoch, G.gen, opt_G, best_loss, best_model, 'generator', args.model_num)
        save_model(args, epoch, D, opt_D, best_loss, best_model, 'discriminator', args.model_num)


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
    train(args)

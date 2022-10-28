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
from mail import send_mail
from evaluation_scripts.cfid.embeddings import InceptionEmbedding
from evaluation_scripts.fid.fid_metric import FIDMetric
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
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), data_range=maxval, multichannel=True
    )

    return ssim


def compute_gradient_penalty(D, real_samples, fake_samples, args, y):
    """Calculates the gradient penalty loss for WGAN GP"""
    Tensor = torch.FloatTensor
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(args.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(input=interpolates, label=y)
    # fake = Tensor(real_samples.shape[0], 1, d_interpolates.shape[-1], d_interpolates.shape[-1]).fill_(1.0).to(args.device)

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


def generate_gif(args, type, ind):
    images = []
    for i in range(8):
        images.append(iio.imread(f'gif_{type}_{i}.png'))

    iio.mimsave(f'variation_gif_{ind}.gif', images, duration=0.25)

    for i in range(8):
        os.remove(f'gif_{type}_{i}.png')
######################

def train(args):
    inception_embedding = InceptionEmbedding(parallel=True)

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
            x, y, mean, std, mask = data[0]
            y = y.cuda()
            x = x.cuda()
            mask = mask.cuda()
            mean = mean.cuda()
            std = std.cuda()

            for j in range(args.num_iters_discriminator):
                for param in D.parameters():
                    param.grad = None

                x_hat = G(y, x=x, mask=mask, truncation=1)

                real_pred = D(input=x, label=y)
                fake_pred = D(input=x_hat, label=y)

                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(D, x.data, x_hat.data, args, y.data)
                real_pred_sq = real_pred ** 2

                d_loss = fake_pred.mean() - real_pred.mean() + 10 * gradient_penalty + 0.001 * real_pred_sq.mean()

                d_loss.backward()
                opt_D.step()

            for param in G.gen.parameters():
                param.grad = None

            x_hat = G(y, x=x, mask=mask, truncation=None)
            fake_pred = D(input=x_hat, label=y)

            g_loss = - fake_pred.mean()

            g_loss.backward()
            opt_G.step()

            batch_loss['g_loss'].append(g_loss.item())
            batch_loss['d_loss'].append(d_loss.item())

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                % (epoch + 1, args.num_epochs, i, len(train_loader.dataset) / args.batch_size, d_loss.item(),
                   g_loss.item())
            )


        print("GETTING DATA LOADERS")
        best_model = False

        if epoch > 50:
            fid_metric = FIDMetric(gan=G,
                                   ref_loader=train_loader,
                                   loader=dev_loader,
                                   image_embedding=inception_embedding,
                                   condition_embedding=inception_embedding,
                                   cuda=True,
                                   args=args,
                                   truncation=None,
                                   truncation_latent=None)

            fid = fid_metric.get_fid()

            best_model = fid < best_loss
            best_loss = fid if best_model else best_loss

            send_mail(f"EPOCH {epoch + 1} UPDATE", f"Metrics:\nFID {fid:.2f}")

        GLOBAL_LOSS_DICT['g_loss'].append(np.mean(batch_loss['g_loss']))
        GLOBAL_LOSS_DICT['d_loss'].append(np.mean(batch_loss['d_loss']))

        save_str = f"END OF EPOCH {epoch + 1}: [Average D loss: {GLOBAL_LOSS_DICT['d_loss'][epoch - start_epoch]:.4f}] [Average G loss: {GLOBAL_LOSS_DICT['g_loss'][epoch - start_epoch]:.4f}]\n"
        print(save_str)

        save_model(args, epoch, G.gen, opt_G, best_loss, best_model, 'generator', 0)
        save_model(args, epoch, D, opt_D, best_loss, best_model, 'discriminator', 0)


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

    try:
        train(args)
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        print(e)
        send_mail("TRAINING CRASH", "Log in to see cause.")

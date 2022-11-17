# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import torch

import numpy as np
import sigpy as sp

from tqdm import tqdm
from mail import send_mail

def symmetric_matrix_square_root_torch(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    u, s, v = torch.linalg.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = s
    si[torch.where(si >= eps)] = torch.sqrt(si[torch.where(si >= eps)])

    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return torch.matmul(torch.matmul(u, torch.diag(si)), v)


def trace_sqrt_product_torch(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.
    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
      => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
      => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
      => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                    = sum(sqrt(eigenvalues(A B B A)))
                                    = sum(eigenvalues(sqrt(A B B A)))
                                    = trace(sqrt(A B B A))
                                    = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.
    Args:
      sigma: a square, symmetric, real, positive semi-definite covariance matrix
      sigma_v: same as sigma
    Returns:
      The trace of the positive square root of sigma*sigma_v
    """

    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = symmetric_matrix_square_root_torch(sigma)

    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))

    return torch.trace(symmetric_matrix_square_root_torch(sqrt_a_sigmav_a))


# **Estimators**
#
def sample_covariance_torch(a, b):
    '''
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    '''
    assert (a.shape[0] == b.shape[0])
    assert (a.shape[1] == b.shape[1])
    m = a.shape[1]
    N = a.shape[0]
    return torch.matmul(torch.transpose(a, 0, 1), b) / N


class CFIDMetric:
    """Helper function for calculating CFID metric.

    Note: This code is adapted from Facebook's FJD implementation in order to compute
    CFID in a streamlined fashion.

    Args:
        gan: Model that takes in a conditioning tensor and yields image samples.
        reference_loader: DataLoader that yields (images, conditioning) pairs
            to be used as the reference distribution.
        condition_loader: Dataloader that yields (image, conditioning) pairs.
            Images are ignored, and conditions are fed to the GAN.
        image_embedding: Function that takes in 4D [B, 3, H, W] image tensor
            and yields 2D [B, D] embedding vectors.
        condition_embedding: Function that takes in conditioning from
            condition_loader and yields 2D [B, D] embedding vectors.
        reference_stats_path: File path to save precomputed statistics of
            reference distribution. Default: current directory.
        save_reference_stats: Boolean indicating whether statistics of
            reference distribution should be saved. Default: False.
        samples_per_condition: Integer indicating the number of samples to
            generate for each condition from the condition_loader. Default: 1.
        cuda: Boolean indicating whether to use GPU accelerated FJD or not.
              Default: False.
        eps: Float value which is added to diagonals of covariance matrices
             to improve computational stability. Default: 1e-6.
    """

    def __init__(self,
                 gan,
                 loader,
                 image_embedding,
                 condition_embedding,
                 cuda=False,
                 args=None,
                 eps=1e-6,
                 num_samps=1):

        self.gan = gan
        self.args = args
        self.loader = loader
        self.image_embedding = image_embedding
        self.condition_embedding = condition_embedding
        self.cuda = cuda
        self.eps = eps
        self.gen_embeds, self.cond_embeds, self.true_embeds = None, None, None
        self.num_samps = num_samps

    def _get_mvue(self, kspace, s_maps):
        return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1) / np.sqrt(
            np.sum(np.square(np.abs(s_maps)), axis=1))

    def _get_embed_im(self, inp):
        embed_ims = torch.zeros(size=(1, 3, 256, 256),
                                device=self.args.device)
        im = 2 * (inp - torch.min(inp)) / (torch.max(inp) - torch.min(inp)) - 1
        embed_ims[0, :, :, :] = im

        return embed_ims

    def _get_generated_distribution(self):
        image_embed = []
        cond_embed = []
        true_embed = []
        count = 0

        for i in range(252):
            print(i)
            with torch.no_grad():
                # for j in range(1):
                # recon_object = torch.load(f'/storage/celebA-HQ/langevin_recons_256/image_{i}_sample_{j}.pt')
                image = torch.zeros(100, 3, 256, 256).cuda()
                true_im = torch.zeros(100, 3, 256, 256).cuda()
                condition_im = torch.zeros(100, 3, 256, 256).cuda()
                for j in range(100):
                    print(j)
                    recon_object = torch.load(f'/storage/celebA-HQ/langevin_recons_256_2/image_{count}.pt')
                    print(recon_object)
                    x_hat[j] = self._get_embed_im(recon_object['x_hat'].cuda())
                    x[j] = self._get_embed_im(recon_object['gt'].cuda())
                    y[j] = self._get_embed_im(recon_object['masked'].cuda())
                    count += 1
                # image = self._get_embed_im(x_hat)
                # condition_im = self._get_embed_im(y)
                # true_im = self._get_embed_im(x)

                img_e = self.image_embedding(image)
                cond_e = self.condition_embedding(condition_im)
                true_e = self.image_embedding(true_im)

                if self.cuda:
                    true_embed.append(true_e.to('cuda:2'))
                    image_embed.append(img_e.to('cuda:1'))
                    cond_embed.append(cond_e.to('cuda:1'))
                else:
                    true_embed.append(true_e.cpu().numpy())
                    image_embed.append(img_e.cpu().numpy())
                    cond_embed.append(cond_e.cpu().numpy())

        if self.cuda:
            true_embed = torch.cat(true_embed, dim=0)
            image_embed = torch.cat(image_embed, dim=0)
            cond_embed = torch.cat(cond_embed, dim=0)
        else:
            true_embed = np.concatenate(true_embed, axis=0)
            image_embed = np.concatenate(image_embed, axis=0)
            cond_embed = np.concatenate(cond_embed, axis=0)

        return image_embed.to(dtype=torch.float64), cond_embed.to(dtype=torch.float64), true_embed.to(
            dtype=torch.float64)

    def get_cfid_torch(self, resample=True):
        y_predict, x_true, y_true = self._get_generated_distribution()

        # mean estimations
        y_true = y_true.to(x_true.device)
        m_y_predict = torch.mean(y_predict, dim=0)
        m_x_true = torch.mean(x_true, dim=0)
        m_y_true = torch.mean(y_true, dim=0)

        no_m_y_true = y_true - m_y_true
        no_m_y_pred = y_predict - m_y_predict
        no_m_x_true = x_true - m_x_true

        m_dist = torch.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)

        u, s, vh = torch.linalg.svd(no_m_x_true.t(), full_matrices=False)
        v = vh.t()
        c_dist_1 = torch.norm(torch.matmul(no_m_y_true.t() - no_m_y_pred.t(), v)) ** 2 / y_true.shape[0]

        v_t_v = torch.matmul(v, vh)
        y_pred_w_v_t_v = torch.matmul(no_m_y_pred.t(), torch.matmul(v_t_v, no_m_y_pred))
        y_true_w_v_t_v = torch.matmul(no_m_y_true.t(), torch.matmul(v_t_v, no_m_y_true))

        c_y_true_given_x_true = 1 / y_true.shape[0] * (torch.matmul(no_m_y_true.t(), no_m_y_true) - y_true_w_v_t_v)
        c_y_predict_given_x_true = 1 / y_true.shape[0] * (torch.matmul(no_m_y_pred.t(), no_m_y_pred) - y_pred_w_v_t_v)

        c_dist_2 = torch.trace(c_y_true_given_x_true + c_y_predict_given_x_true) - 2 * trace_sqrt_product_torch(
            c_y_predict_given_x_true, c_y_true_given_x_true)

        cfid = m_dist + c_dist_1 + c_dist_2

        return cfid.cpu().numpy()

    def get_cfid_torch_pinv(self):
        y_predict, x_true, y_true = self._get_generated_distribution()

        # mean estimations
        y_true = y_true.to(x_true.device)
        m_y_predict = torch.mean(y_predict, dim=0)
        m_x_true = torch.mean(x_true, dim=0)
        m_y_true = torch.mean(y_true, dim=0)

        no_m_y_true = y_true - m_y_true
        no_m_y_pred = y_predict - m_y_predict
        no_m_x_true = x_true - m_x_true

        c_y_predict_x_true = torch.matmul(no_m_y_pred.t(), no_m_x_true) / y_predict.shape[0]
        c_y_predict_y_predict = torch.matmul(no_m_y_pred.t(), no_m_y_pred) / y_predict.shape[0]
        c_x_true_y_predict = torch.matmul(no_m_x_true.t(), no_m_y_pred) / y_predict.shape[0]

        c_y_true_x_true = torch.matmul(no_m_y_true.t(), no_m_x_true) / y_predict.shape[0]
        c_x_true_y_true = torch.matmul(no_m_x_true.t(), no_m_y_true) / y_predict.shape[0]
        c_y_true_y_true = torch.matmul(no_m_y_true.t(), no_m_y_true) / y_predict.shape[0]

        inv_c_x_true_x_true = torch.linalg.pinv(torch.matmul(no_m_x_true.t(), no_m_x_true) / y_predict.shape[0])

        c_y_true_given_x_true = c_y_true_y_true - torch.matmul(c_y_true_x_true,
                                                            torch.matmul(inv_c_x_true_x_true, c_x_true_y_true))
        c_y_predict_given_x_true = c_y_predict_y_predict - torch.matmul(c_y_predict_x_true,
                                                                     torch.matmul(inv_c_x_true_x_true, c_x_true_y_predict))
        c_y_true_x_true_minus_c_y_predict_x_true = c_y_true_x_true - c_y_predict_x_true
        c_x_true_y_true_minus_c_x_true_y_predict = c_x_true_y_true - c_x_true_y_predict

        # Distance between Gaussians
        m_dist = torch.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
        c_dist1 = torch.trace(torch.matmul(torch.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_c_x_true_x_true),
                                            c_x_true_y_true_minus_c_x_true_y_predict))
        c_dist_2_1 = torch.trace(c_y_true_given_x_true + c_y_predict_given_x_true)
        c_dist_2_2 = - 2 * trace_sqrt_product_torch(
            c_y_predict_given_x_true, c_y_true_given_x_true)

        c_dist2 = c_dist_2_1 + c_dist_2_2

        c_dist = c_dist1 + c_dist2

        print(f"M: {m_dist.cpu().numpy()}")
        print(f"C: {c_dist.cpu().numpy()}")

        send_mail("LANGEVIN CFID RESULTS", f"M_DIST: {m_dist.cpu().numpy()}", f"C_DIST: {c_dist.cpu().numpy()}")

        cfid = m_dist + c_dist1 + c_dist2

        return cfid.cpu().numpy()

import pathlib
import shutil
import torch
import numpy as np


# THIS FILE CONTAINTS UTILITY FUNCTIONS FOR OUR GAN AND A WRAPPER CLASS FOR THE GENERATOR
def load_best_gan(args, model_num=0):
    from utils.prepare_models import build_model
    checkpoint_file_gen = pathlib.Path(
        f'{args.checkpoint_dir}/generator_best_model.pt')
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

    generator = build_model(args)
    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(checkpoint_gen['model'])

    generator = GANWrapper(generator, args)

    return generator


def get_gan(args):
    from utils.prepare_models import build_model, build_optim, build_discriminator

    if args.resume:
        checkpoint_file_gen = pathlib.Path(
            f'{args.checkpoint_dir}/generator_model.pt')
        checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

        checkpoint_file_dis = pathlib.Path(
            f'{args.checkpoint_dir}/discriminator_model.pt')
        checkpoint_dis = torch.load(checkpoint_file_dis, map_location=torch.device('cuda'))

        generator = build_model(args)
        discriminator = build_discriminator(args)

        if args.data_parallel:
            generator = torch.nn.DataParallel(generator)
            discriminator = torch.nn.DataParallel(discriminator)

        generator.load_state_dict(checkpoint_gen['model'])

        generator = GANWrapper(generator, args)

        opt_gen = build_optim(args, generator.gen.parameters())
        opt_gen.load_state_dict(checkpoint_gen['optimizer'])

        discriminator.load_state_dict(checkpoint_dis['model'])

        opt_dis = build_optim(args, discriminator.parameters())
        opt_dis.load_state_dict(checkpoint_dis['optimizer'])

        best_loss = checkpoint_gen['best_dev_loss']
        start_epoch = checkpoint_gen['epoch']

    else:
        generator = build_model(args)
        discriminator = build_discriminator(args)

        if args.data_parallel:
            generator = torch.nn.DataParallel(generator)
            discriminator = torch.nn.DataParallel(discriminator)

        generator = GANWrapper(generator, args)

        # Optimizers
        opt_gen = torch.optim.Adam(generator.gen.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        opt_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        best_loss = 0
        start_epoch = 0

    return generator, discriminator, opt_gen, opt_dis, best_loss, start_epoch


def save_model(args, epoch, model, optimizer, best_dev_loss, is_new_best, m_type, model_dir):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': args.exp_dir
        },
        f=args.exp_dir / f'{m_type}_model.pt'
    )

    if is_new_best:
        shutil.copyfile(args.exp_dir / f'{m_type}_model.pt',
                        args.exp_dir / f'{m_type}_best_model.pt')


class GANWrapper:
    def __init__(self, gen, args):
        self.args = args
        self.resolution = args.im_size
        self.gen = gen

    def get_noise(self, num_vectors, var):
        # return torch.cuda.FloatTensor(np.random.normal(size=(num_vectors, self.args.latent_size), scale=1))
        return torch.empty((num_vectors, 2, self.resolution, self.resolution)).normal_(mean=0, std=np.sqrt(var)).cuda()

    def update_gen_status(self, val):
        self.gen.eval() if val else self.gen.train()

    def data_consistency(self, samples, masked_ims, mask):
        # samples[:, :, inds[0], inds[1]] = masked_ims[:, :, inds[0], inds[1]]
        return samples * (1 - mask) + masked_ims

    def get_mean_code_vector(self, y, x, mask, num_latents=32):
        mean_latent = torch.zeros(y.size(0), num_latents, 512)
        for z in range(num_latents):
            mean_latent[:, z, :] = self.gen(x, mask, [torch.randn(y.size(0), 512, device=y.device)], return_latents=True, truncation=None, truncation_latent=None)[:, 0, :]

        return torch.mean(mean_latent, dim=1)

    def __call__(self, y, noise_var=1, x=None, mask=None, truncation=None, truncation_latent=None):
        # num_vectors = y.size(0)
        # z = self.get_noise(num_vectors, noise_var)
        # samples = self.gen(y)
        samples = self.gen(x, mask, [torch.randn(y.size(0), 512, device=y.device)], return_latents=False, truncation=truncation, truncation_latent=truncation_latent)
        # samples = self.data_consistency(samples, y, mask)

        return samples

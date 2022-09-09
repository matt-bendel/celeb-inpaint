import torch

from models.generators.our_gen import GeneratorModel
from models.generators.stylegan import StyleGAN
from models.discriminators.patch_disc_2 import PatchDisc
from models.discriminators.unet_disc import Unet_Discriminator


def build_model(args):
    # model = GeneratorModel(
    #     in_chans=args.in_chans + 2,
    #     out_chans=args.out_chans,
    #     latent_size=args.latent_size
    # ).to(torch.device('cuda'))
    model = StyleGAN().to(torch.device('cuda'))

    return model


def build_discriminator(args):
    model = PatchDisc(
        # input_nc=args.in_chans*2
    ).to(torch.device('cuda'))

    return model


def build_optim(args, params):
    return torch.optim.Adam(params, lr=args.lr, betas=(args.beta_1, args.beta_2))

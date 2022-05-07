import cv2
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as F

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, args):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create  a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.args = args
        np.random.seed(0)

        total = 128 * 128
        n = total // self.args.R

        arr = np.zeros(total)
        arr[:n] = 1
        np.random.shuffle(arr)
        self.mask = np.reshape(arr, (128, 128))
        self.inv_mask = 1 - self.mask

    def __call__(self, gt_im):
        print(gt_im.shape)
        exit()
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        gt = (gt_im - mean[:, None, None]) / std[:, None, None]
        masked_im = gt * self.mask[None, :, :]

        inds = np.where(self.inv_mask == 1)

        return gt, masked_im, mean, std, inds



def create_datasets(args):
    transform = transforms.Compose([transforms.ToTensor(), DataTransform(args)])
    dataset = datasets.ImageFolder('/storage/celebA-HQ/celeba_hq_128', transform=transform)
    train_data, dev_data, test_data = torch.utils.data.random_split(
        dataset, [24000, 4000, 2000],
        generator=torch.Generator().manual_seed(0)
    )

    return test_data, dev_data, train_data


def create_data_loaders(args):
    test_data, dev_data, train_data = create_datasets(args)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, dev_loader, test_loader

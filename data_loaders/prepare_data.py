import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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
        # n = total // self.args.R

        arr = np.ones((128, 128))
        arr[128 // 4: 3 *128//4, 128 // 4: 3 *128//4] = 0
        plt.imshow(np.reshape(arr, (128, 128)), cmap='viridis')
        plt.savefig(f'mask_{self.args.R}.png')
        plt.close()
        self.mask = torch.tensor(np.reshape(arr, (128, 128)), dtype=torch.float).repeat(3, 1, 1)
        torch.save(self.mask, 'mast.pt')

    def __call__(self, gt_im):
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        gt = (gt_im - mean[:, None, None]) / std[:, None, None]
        masked_im = gt * self.mask

        return gt, masked_im, mean, std, self.mask



def create_datasets(args):
    transform = transforms.Compose([transforms.ToTensor(), DataTransform(args)])
    dataset = datasets.ImageFolder('/storage/celebA-HQ/celeba_hq_128', transform=transform)
    train_data, dev_data, test_data = torch.utils.data.random_split(
        dataset, [27000, 2000, 1000],
        generator=torch.Generator().manual_seed(0)
    )

    print(test_data.dataset.imgs)
    exit()

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
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, dev_loader, test_loader

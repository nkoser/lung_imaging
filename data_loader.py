import random

import torch
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset


def normalizeVolumes(npzarray):
    maxHU = npzarray.max()
    minHU = npzarray.min()
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray


class Lung_loader(Dataset):

    def __init__(self, down_paths, org_paths, transform=None, patch_size=(64, 64, 64)):
        """
        :param down_paths: path to the downsample CT-Scans
        :param org_paths: path to the original CT-Scans
        """

        self.down_paths = down_paths
        self.org_paths = org_paths
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.down_paths)

    def __getitem__(self, idx):
        down_img_path = self.down_paths[idx]
        org_img_path = self.org_paths[idx]
        down_img = torch.tensor(nib.load(down_img_path).get_fdata())
        org_img = torch.tensor(nib.load(org_img_path).get_fdata())

        print(down_img_path, org_img_path)

        x_offset, y_offset, z_offset = (random.randint(-60, 60), random.randint(-60, 60), random.randint(-60, 60))

        voxel_coord = (
            org_img.shape[0] // 2 - x_offset, org_img.shape[1] // 2 - y_offset, org_img.shape[2] // 2 - z_offset)

        print(voxel_coord)

        img_patch = org_img[
                    int(voxel_coord[0] - self.patch_size[0] / 2):int(voxel_coord[0] + self.patch_size[0] / 2),
                    int(voxel_coord[1] - self.patch_size[1] / 2):int(voxel_coord[1] + self.patch_size[1] / 2),
                    int(voxel_coord[2] - self.patch_size[2] / 2):int(voxel_coord[2] + self.patch_size[2] / 2)]

        down_patch = down_img[
                     int(voxel_coord[0] - self.patch_size[0] / 2):int(voxel_coord[0] + self.patch_size[0] / 2),
                     int(voxel_coord[1] - self.patch_size[1] / 2):int(voxel_coord[1] + self.patch_size[1] / 2),
                     int(voxel_coord[2] - self.patch_size[2] / 2):int(voxel_coord[2] + self.patch_size[2] / 2)]

        img_patch = F.pad(input=img_patch, pad=(0, abs(self.patch_size[2] - img_patch.shape[2]),
                                                0, abs(self.patch_size[1] - img_patch.shape[1]),
                                                0, abs(self.patch_size[0] - img_patch.shape[0])),
                          mode='constant',
                          value=0)
        down_patch = F.pad(input=down_patch, pad=(0, abs(self.patch_size[2] - down_patch.shape[2]),
                                                  0, abs(self.patch_size[1] - down_patch.shape[1]),
                                                  0, abs(self.patch_size[0] - down_patch.shape[0])),
                           mode='constant',
                           value=0)

        if self.transform:
            pass

        return down_patch.unsqueeze(0), img_patch.unsqueeze(0)

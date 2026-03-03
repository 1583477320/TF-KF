import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import random
import torchvision.transforms.functional as F

import os
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = 299
NUM_JOINTS = 17
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RandomTFColorDistortion:
    """
    快速版颜色扰动
    """
    def __init__(self, fast_mode=True):
        self.fast_mode = fast_mode

    def __call__(self, x):
        # x: Tensor [3,H,W], range [0,1]
        if torch.rand(1) < 0.5:
            brightness = (torch.rand(1).item() - 0.5) * (32 / 255.0)
            x = x + brightness

        if torch.rand(1) < 0.5:
            sat = 0.5 + torch.rand(1).item() * 1.0
            mean = x.mean(dim=0, keepdim=True)
            x = (x - mean) * sat + mean

        return torch.clamp(x, 0.0, 1.0)


def build_inception_train_transform():
    return transforms.Compose([
        transforms.Resize(320),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ToTensor(),                 # [0,1]
        RandomTFColorDistortion(fast_mode=True),
        transforms.Normalize(mean=[0.5,0.5,0.5],
                             std=[0.5,0.5,0.5])  # -> [-1,1]
    ])


def build_inception_test_transform():
    return transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],
                             std=[0.5,0.5,0.5])
    ])


class H36MDataset(Dataset):
    def __init__(self, img_txt, h5_path, img_root, transform=None):
        self.h5_path = h5_path
        self.img_root = img_root
        self.transform = transform

        self.h5_file = h5py.File(self.h5_path, "r")
        self.S = self.h5_file["S"][:]            # (N,17,3)
        self.img_names = self._load_txt(img_txt)
        self.img_paths = [Path(img_root) / n for n in self.img_names]

        self.length = self.S.shape[0]
        print("Dataset size:", self.length)

    def _load_txt(self, path):
        with open(path) as f:
            return [l.strip() for l in f if l.strip()]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # ---------- label ----------
        pose3d = self.S[idx].astype(np.float32)      # (17,3)

        # 可选：尺度归一化（mm -> m）
        pose3d = pose3d / 1000.0

        pose3d = torch.from_numpy(pose3d).view(-1)   # (51,)

        # ---------- image ----------
        image = Image.open(self.img_paths[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, pose3d


class H36MSequenceDataset(Dataset):
    def __init__(self, h5_path, seq_length=10, step=5, mode="train"):
        self.seq_length = seq_length
        self.step = step
        self.mode = mode

        with h5py.File(h5_path, 'r') as f:
            self.data_2d = f['part'][:].reshape(-1, 34).astype(np.float32)
            self.data_3d = f['S'][:].reshape(-1, 51).astype(np.float32)

        # ========= Normalization =========
        if mode == "train":
            self.mean2d = self.data_2d.mean(axis=0)
            self.std2d  = self.data_2d.std(axis=0) + 1e-6
            self.mean3d = self.data_3d.mean(axis=0)
            self.std3d  = self.data_3d.std(axis=0) + 1e-6

            np.savez(
                "seq_norm_stats.npz",
                mean2d=self.mean2d,
                std2d=self.std2d,
                mean3d=self.mean3d,
                std3d=self.std3d
            )
        else:
            stats = np.load("seq_norm_stats.npz")
            self.mean2d, self.std2d = stats["mean2d"], stats["std2d"]
            self.mean3d, self.std3d = stats["mean3d"], stats["std3d"]

        # Normalize
        self.data_2d = (self.data_2d - self.mean2d) / self.std2d
        self.data_3d = (self.data_3d - self.mean3d) / self.std3d

        if mode == 'train':
            np.savez(
                "norm_stats.npz",
                mean2d=self.mean2d,
                std2d=self.std2d,
                mean3d=self.mean3d,
                std3d=self.std3d
            )
        # ========= Build valid sequence indices =========
        max_start = len(self.data_2d) - seq_length
        self.valid_indices = list(range(0, max_start, step))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.seq_length

        # (Seq, 34)
        sample_2d = self.data_2d[start:end]

        # (51,)
        target_3d = self.data_3d[end - 1]

        return (
            torch.from_numpy(sample_2d).float(),
            torch.from_numpy(target_3d).float()
        )

import os
from os import path

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    InterpolationMode,
)
from torchvision.transforms import (
    functional as TF,
)

from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple


class TwoAFCDataset(Dataset):
    # Berkeley Adobe Perceptual Patch Similarity
    # 2AFC dataset
    def __init__(self, dataset_dir, training, load_all=False):
        if load_all:
            dataset_dirs = [path.join(dataset_dir, "train"), path.join(dataset_dir, "val")]
        else:
            if training:
                dataset_dirs = [path.join(dataset_dir, "train")]
            else:
                dataset_dirs = [path.join(dataset_dir, "val")]

        self.triplets = []
        self.judges = []

        for dataset_dir in dataset_dirs:
            for subdir in os.listdir(dataset_dir):
                subdir = path.join(dataset_dir, subdir)
                if not path.isdir(subdir):
                    continue
                refs = ImageLoader.listdir(path.join(subdir, "ref"))
                for ref in refs:
                    basename = path.basename(ref)
                    p0 = path.join(subdir, "p0", basename)
                    p1 = path.join(subdir, "p1", basename)
                    judge = path.join(subdir, "judge", path.splitext(basename)[0] + ".npy")

                    if all(path.exists(fn) for fn in (ref, p0, p1, judge)):
                        self.triplets.append((ref, p0, p1))
                        self.judges.append(judge)

    def __len__(self):
        return len(self.judges)

    def create_sampler(self, num_samples):
        return torch.utils.data.sampler.RandomSampler(
            self,
            num_samples=num_samples,
            replacement=True)

    @staticmethod
    def load_resized(filepath):
        im, _ = load_image_simple(filepath, color="rgb")
        if im.size == (256, 256):
            # Downscaling without interpolation because the image is 4x NEAREST upscaling
            return TF.resize(im, 64, interpolation=InterpolationMode.NEAREST_EXACT)
        else:
            # Very few images are not 256x256. I don't know what that is, so use the same method as the original
            return TF.resize(im, 64, interpolation=T.InterpolationMode.BILINEAR)

    def __getitem__(self, index):
        triplet = self.triplets[index]
        ref = self.load_resized(triplet[0])
        p0 = self.load_resized(triplet[1])
        p1 = self.load_resized(triplet[2])
        judge = torch.from_numpy(np.load(self.judges[index])).reshape(1, 1, 1).float()

        ref = TF.to_tensor(ref)
        p0 = TF.to_tensor(p0)
        p1 = TF.to_tensor(p1)

        # Normalization should be performed on the model side if necessary.
        # judge is [0, 1] range, not [-1, 1].

        return ref, p0, p1, judge


def _test_dataset(dataset_dir):
    from tqdm import tqdm

    dataset_train = TwoAFCDataset(dataset_dir, train=True)
    dataset_test = TwoAFCDataset(dataset_dir, train=False)
    print("train", len(dataset_train), "test", len(dataset_test))

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
    )
    for ref, p0, p1, judge in tqdm(test_loader, ncols=80):
        pass


if __name__ == "__main__":
    _test_dataset("data/2afc")

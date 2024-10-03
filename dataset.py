from typing import Literal
from torch.utils.data import Subset
from tqdm import tqdm
from torchbench.datasets import ADE20K
from torchvision.transforms.functional import (
    to_tensor,
    pil_to_tensor,
    resize,
    normalize,
)
from torchvision.transforms import InterpolationMode



class ADE20K_Dataset:
    @staticmethod
    def __transforms(image, target):
        image = resize(to_tensor(image), (256, 256))
        image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        target = resize(pil_to_tensor(target), (64, 64), InterpolationMode.NEAREST)[
            0
        ].long()
        # 0: background, -1: plant, -2: person/animal, -3: vehicle
        class_mapping = [
            (5, -1),
            (10, -1),
            (13, -2),
            (18, -1),
            (21, -3),
            (67, -1),
            (77, -3),
            (84, -3),
            (91, -3),
            (103, -3),
            (104, -3),
            (127, -2),
            (128, -3),
        ]
        for cm in class_mapping:
            target[target == cm[0]] = cm[1]
        target[target > 0] = 0
        target *= -1
        return image, target

    @staticmethod
    def __download(type: Literal["train", "test", "val"]):
        try:
            dataset = ADE20K("data", "train", transforms=ADE20K_Dataset.__transforms)
        except FileNotFoundError:
            dataset = ADE20K(
                "data", "train", download=True, transforms=ADE20K_Dataset.__transforms
            )

        return dataset

    @staticmethod
    def __preprocess(dataset):
        valid_indices = []
        print('Trimming dataset')
        for entry in tqdm(range(len(dataset))):
            _, target = dataset[entry]

            if (target > 0).float().mean() > 0.01:
                valid_indices.append(entry)

        if len(dataset) > len(valid_indices):
            dataset = Subset(dataset, valid_indices)

        return dataset

    @staticmethod
    def train_dataset():
        train_dataset = ADE20K_Dataset.__download("train")
        train_dataset = ADE20K_Dataset.__preprocess(train_dataset)

        return train_dataset

    @staticmethod
    def test_dataset():
        train_dataset = ADE20K_Dataset.__download("test")
        train_dataset = ADE20K_Dataset.__preprocess(train_dataset)

        return train_dataset

    @staticmethod
    def val_dataset():
        train_dataset = ADE20K_Dataset.__download("val")
        train_dataset = ADE20K_Dataset.__preprocess(train_dataset)

        return train_dataset

Datasets = {
    'ADE20K': ADE20K_Dataset
}
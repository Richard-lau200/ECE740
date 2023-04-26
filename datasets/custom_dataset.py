# from __future__ import division

# import json
# import logging

# import numpy as np
# import torchvision.transforms as transforms
# from PIL import Image
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data.sampler import RandomSampler

# from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
# from datasets.image_reader import build_image_reader
# from datasets.transforms import RandomColorJitter

# logger = logging.getLogger("global_logger")


# def build_custom_dataloader(cfg, training, distributed=True):

#     image_reader = build_image_reader(cfg.image_reader)

#     normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
#     if training:
#         transform_fn = TrainBaseTransform(
#             cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
#         )
#     else:
#         transform_fn = TestBaseTransform(cfg["input_size"])

#     colorjitter_fn = None
#     if cfg.get("colorjitter", None) and training:
#         colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

#     logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))

#     dataset = CustomDataset(
#         image_reader,
#         cfg["meta_file"],
#         training,
#         transform_fn=transform_fn,
#         normalize_fn=normalize_fn,
#         colorjitter_fn=colorjitter_fn,
#     )

#     if distributed:
#         sampler = DistributedSampler(dataset)
#     else:
#         sampler = RandomSampler(dataset)

#     data_loader = DataLoader(
#         dataset,
#         batch_size=cfg["batch_size"],
#         num_workers=cfg["workers"],
#         pin_memory=True,
#         sampler=sampler,
#     )

#     return data_loader


# class CustomDataset(BaseDataset):
#     def __init__(
#         self,
#         image_reader,
#         meta_file,
#         training,
#         transform_fn,
#         normalize_fn,
#         colorjitter_fn=None,
#     ):
#         self.image_reader = image_reader
#         self.meta_file = meta_file
#         self.training = training
#         self.transform_fn = transform_fn
#         self.normalize_fn = normalize_fn
#         self.colorjitter_fn = colorjitter_fn

#         # construct metas
#         with open(meta_file, "r") as f_r:
#             self.metas = []
#             for line in f_r:
#                 meta = json.loads(line)
#                 self.metas.append(meta)

#     def __len__(self):
#         return len(self.metas)

#     def __getitem__(self, index):
#         input = {}
#         meta = self.metas[index]

#         # read image
#         filename = meta["filename"]
#         label = meta["label"]
#         image = self.image_reader(meta["filename"])
#         input.update(
#             {
#                 "filename": filename,
#                 "height": image.shape[0],
#                 "width": image.shape[1],
#                 "label": label,
#             }
#         )

#         if meta.get("clsname", None):
#             input["clsname"] = meta["clsname"]
#         else:
#             input["clsname"] = filename.split("/")[-4]

#         image = Image.fromarray(image, "RGB")

#         # read / generate mask
#         if meta.get("maskname", None):
#             mask = self.image_reader(meta["maskname"], is_mask=True)
#         else:
#             if label == 0:  # good
#                 mask = np.zeros((image.height, image.width)).astype(np.uint8)
#             elif label == 1:  # defective
#                 mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
#             else:
#                 raise ValueError("Labels must be [None, 0, 1]!")

#         mask = Image.fromarray(mask, "L")

#         if self.transform_fn:
#             image, mask = self.transform_fn(image, mask)
#         if self.colorjitter_fn:
#             image = self.colorjitter_fn(image)
#         image = transforms.ToTensor()(image)
#         mask = transforms.ToTensor()(mask)
#         if self.normalize_fn:
#             image = self.normalize_fn(image)
#         input.update({"image": image, "mask": mask})
#         return input
# from __future__ import division

# import json
# import logging

# import numpy as np
# import torchvision.transforms as transforms
# from PIL import Image
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data.sampler import RandomSampler

# from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
# from datasets.image_reader import build_image_reader
# from datasets.transforms import RandomColorJitter

# logger = logging.getLogger("global_logger")


# def build_custom_dataloader(cfg, training, distributed=True):
#     print(cfg) 
#     image_reader = build_image_reader(cfg.image_reader)

#     normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
#     if training:
#         transform_fn = TrainBaseTransform(
#             cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
#         )
#     else:
#         transform_fn = TestBaseTransform(cfg["input_size"])

#     colorjitter_fn = None
#     if cfg.get("colorjitter", None) and training:
#         colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

#     logger.info("building CIFAR100 Dataset from: {}".format(cfg["meta_file"]))

#     dataset = CIFAR100Dataset(
#         image_reader,
#         cfg["meta_file"],
#         training,
#         transform_fn=transform_fn,
#         normalize_fn=normalize_fn,
#         colorjitter_fn=colorjitter_fn,
#     )

#     if distributed:
#         sampler = DistributedSampler(dataset)
#     else:
#         sampler = RandomSampler(dataset)

#     data_loader = DataLoader(
#         dataset,
#         batch_size=cfg["batch_size"],
#         num_workers=cfg["workers"],
#         pin_memory=True,
#         sampler=sampler,
#     )

#     return data_loader


# class CIFAR100Dataset(BaseDataset):
#     base_folder = "cifar-100-python"
#     url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
#     filename = "cifar-100-python.tar.gz"
#     tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
#     train_list = [
#         ["train", "16019d7e3df5f24257cddd939b257f8d"],
#     ]
#     test_list = [
#         ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
#     ]
#     meta = {
#         "filename": "meta",
#         "md5": "5ff9c542aee3614f3951f8cda6e48888",
#     }

#     def __init__(
#         self,
#         image_reader,
#         meta_file,
#         training,
#         transform_fn,
#         normalize_fn,
#         colorjitter_fn=None,
#     ):
#         self.image_reader = image_reader
#         self.meta_file = meta_file
#         self.training = training
#         self.transform_fn = transform_fn
#         self.normalize_fn = normalize_fn
#         self.colorjitter_fn = colorjitter_fn

#         # Load CIFAR-100 data
#         if self.training:
#             downloaded_list = self.train_list
#         else:
#             downloaded_list = self.test_list

#         self.data = []
#         self.targets = []

#         for file_name, checksum in downloaded_list:
#             file_path = os.path.join(self.root, self.base_folder, file_name)
#             with open(file_path, "rb") as f:
#                 entry = pickle.load(f, encoding="latin1")
#                 self.data.append(entry["data"])
#                 if "labels" in entry:
#                     self.targets.extend(entry["labels"])
#                 else:
#                     self.targets.extend(entry["fine_labels"])

#         self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
#         self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC

#         # Load metadata
#         path = os.path.join(self.root, self.base_folder, self.meta["filename"])
#         if not check_integrity(path, self.meta["md5"]):
#             raise RuntimeError("Dataset metadata file not found or corrupted.")
#         with open(path, "rb") as infile:
#             meta = pickle.load(infile, encoding="latin1")
#         self.classes = meta["fine_label_names"]

#     def __len__(self):
#         return len(self.targets)

#     def __getitem__(self, index):
#         input = {}
#         image = self.data[index]
#         label = self.targets[index]

#         input.update(
#             {
#                 "filename": f"cifar100_{index}.png",
#                 "height": image.shape[0],
#                 "width": image.shape[1],
#                 "label": label,
#                 "clsname": self.classes[label],
#             }
#         )

#         image = Image.fromarray(image, "RGB")

#         if self.transform_fn:
#             image = self.transform_fn(image)
#         if self.colorjitter_fn:
#             image = self.colorjitter_fn(image)
#         image = transforms.ToTensor()(image)
#         if self.normalize_fn:
#             image = self.normalize_fn(image)
#         input.update({"image": image})
#         return input
from __future__ import division

import logging
import os.path
import pickle
import random
from typing import Any, List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

logger = logging.getLogger("global_logger")

def build_custom_dataloader(cfg, training, distributed=True):
    logger.info("building CustomDataset from: {}".format(cfg["root_dir"]))

    dataset = CIFAR100(
        root=cfg["root_dir"],
        train=training,
        resize=cfg["input_size"],
        normals=cfg["normals"],
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader

class CIFAR100(Dataset):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]
    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool,
        resize: List[int],
        normals: List[int],
    ) -> None:

        self.root = root
        self.normals = normals
        self.train = train

        self.transform = transforms.Compose(
            [
                transforms.Resize(resize, Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()
        self._select_normal()

    def _select_normal(self) -> None:
        assert self.data.shape[0] == len(self.targets)
        _data_normal = []
        _data_defect = []
        _targets_normal = []
        _targets_defect = []
        for datum, target in zip(self.data, self.targets):
            if target in self.normals:
                _data_normal.append(datum)
                _targets_normal.append(target)
            elif not self.train:
                _data_defect.append(datum)
                _targets_defect.append(target)

        if not self.train:
            # ids = random.sample(range(len(_data_defect)), len(_data_normal))
            ids = random.sample(range(len(_data_defect)), min(len(_data_normal), len(_data_defect)))
            _data_defect = [_data_defect[idx] for idx in ids]
            _targets_defect = [_targets_defect[idx] for idx in ids]

        self.data = _data_normal + _data_defect
        self.targets = _targets_normal + _targets_defect

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data["fine_label_names"]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        label = 0 if target in self.normals else 1

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        height = img.shape[1]
        width = img.shape[2]

        if label == 0:
            mask = torch.zeros((1, height, width))
        else:
            mask = torch.ones((1, height, width))

        input = {
            "filename": "{}/{}.jpg".format(self.classes[target], index),
            "image": img,
            "mask": mask,
            "height": height,
            "width": width,
            "label": label,
            "clsname": "cifar",
        }

        return input

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


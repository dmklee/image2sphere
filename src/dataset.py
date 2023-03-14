from typing import Optional, List, Callable
import os
import glob
import numpy as np
import torch
import torchvision
from PIL import Image

class ModelNet10Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 train: bool,
                 limited: bool = False,
                ):
        name = f"modelnet10_{'train' if train else 'test'}.npz"
        if limited and train:
            name = name.replace('_', '_limited_')
        path = os.path.join(dataset_path, "modelnet10", name)
        data = np.load(path)

        self.data = {
            'img' : torch.from_numpy(data['imgs']).permute(0, 3, 1, 2),
            'rot' : torch.from_numpy(data['rots']),
            'cls' : torch.from_numpy(data['cat_ids']).unsqueeze(-1).long(),
        }

        self.num_classes = 10
        self.class_names = ('bathtub', 'bed', 'chair', 'desk', 'dresser',
                            'monitor', 'night_stand', 'sofa', 'table', 'toilet')

    def __getitem__(self, index):
        img = self.data['img'][index].to(torch.float32) / 255.

        if img.shape[0] != 3:
            img = img.expand(3,-1,-1)

        class_index = self.data['cls'][index]

        rot = self.data['rot'][index]

        return dict(img=img, cls=class_index, rot=rot)

    @property
    def img_shape(self):
        return (3, 224, 224)

    def __len__(self):
        return len(self.data['img'])


class SymsolDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 train: bool,
                 set_number: int=1,
                 num_views: int=None,
                ):
        self.mode = 'train' if train else 'test'
        self.path = os.path.join(dataset_path, "symsol", self.mode)
        rotations_data = np.load(os.path.join(self.path, 'rotations.npz'))
        self.class_names = {
            1 : ('tet', 'cube', 'icosa', 'cone', 'cyl'),
            2 : ('sphereX',),#, 'cylO', 'sphereX'),
            3 : ('cylO',),#, 'cylO', 'sphereX'),
            4 : ('tetX',),#, 'cylO', 'sphereX'),
        }[set_number]
        self.num_classes = len(self.class_names)

        self.rotations_data = [rotations_data[c][:num_views] for c in self.class_names]
        self.indexers = np.cumsum([len(v) for v in self.rotations_data])

    def __getitem__(self, index):
        cls_ind = np.argmax(index < self.indexers)
        if cls_ind > 0:
            index = index - self.indexers[cls_ind-1]

        rot = self.rotations_data[cls_ind][index]
        # randomly sample one of the valid rotation labels
        rot = rot[np.random.randint(len(rot))]
        rot = torch.from_numpy(rot)

        im_path = os.path.join(self.path, 'images',
                               f'{self.class_names[cls_ind]}_{str(index).zfill(5)}.png')
        img = np.array(Image.open(im_path))
        img = torch.from_numpy(img).to(torch.float32) / 255.
        img = img.permute(2, 0, 1)

        class_index = torch.tensor((cls_ind,), dtype=torch.long)

        return dict(img=img, cls=class_index, rot=rot)

    def __len__(self):
        return self.indexers[-1]

    @property
    def img_shape(self):
        return (3, 224, 224)

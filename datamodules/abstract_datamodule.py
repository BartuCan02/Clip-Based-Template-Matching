import torch
import random
import numpy as np

from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader

from .transforms import get_transforms

# worker_init_fn
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class AbstractDataModule(LightningDataModule):
    def __init__(self, args, datadir='data', batchsize=256, num_workers=0, num_exemplars = 1):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None
        self.dataset_train = self.dataset_val = None
        self.num_exemplars = num_exemplars

        self.sampler = None
        self.collate = None
        self.worker_seed_func = seed_worker
        self.g = torch.Generator()
        self.g.manual_seed(args.seed)
        
        self.args = args

        self.transforms_list = get_transforms(args.image_size)
        self.train_transform = self.transforms_list['default']
        self.val_transform = self.transforms_list['default']

    @property
    def num_classes(self) -> int:
        return None

    def setup(self, stage: str):
        self.dataset_train = self.dataset(root=self.hparams.datadir, transform=self.train_transform)
        self.dataset_val = self.dataset(root=self.hparams.datadir, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers, shuffle=False if self.sampler else True, sampler=self.sampler, collate_fn=self.collate, worker_init_fn=self.worker_seed_func, generator=self.g, drop_last=True)

    def unshuffled_train_dataloader(self):
        if self.dataset_train is None:
            self.setup(stage='init')
        return DataLoader(self.dataset_train, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers, collate_fn=self.collate, worker_init_fn=self.worker_seed_func, generator=self.g)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.hparams.batchsize, num_workers=self.hparams.num_workers, collate_fn=self.collate, worker_init_fn=self.worker_seed_func, generator=self.g)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()

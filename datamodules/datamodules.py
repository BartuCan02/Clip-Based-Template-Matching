import os
from torch.utils.data import DataLoader

from .collate import custom_collate
from .abstract_datamodule import AbstractDataModule
from .datasets.RPINE import RPINE_Dataset
from .datasets.FSCD147 import FSCD147_Dataset
from .datasets.FSCD_LVIS import FSCD_LVIS_Dataset

class RPINEDataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = RPINE_Dataset
        self.collate = custom_collate

        self.train_transform = self.transforms_list['default'] 
        self.val_transform =self.transforms_list['default']

    def setup(self, stage: str):
        traindir = os.path.join(self.hparams.datadir, 'train')
        valdir = os.path.join(self.hparams.datadir, 'val')

        self.dataset_train = RPINE_Dataset(root=traindir, transform=self.train_transform, max_exemplars = self.num_exemplars, scale_factor=32, split="train")
        self.dataset_val = RPINE_Dataset(root=valdir, transform=self.val_transform, max_exemplars = self.num_exemplars, scale_factor=32, split="test", now_eval=self.args.eval)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=1, num_workers=self.hparams.num_workers, collate_fn=self.collate, worker_init_fn=self.worker_seed_func, generator=self.g)
    
class FSCD147DataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = FSCD147_Dataset
        self.collate = custom_collate

        self.train_transform = self.transforms_list['default'] 
        self.val_transform =self.transforms_list['default']

    def setup(self, stage: str):
        traindir = self.hparams.datadir
        valdir = self.hparams.datadir

        self.dataset_train = FSCD147_Dataset(root=traindir, transform=self.train_transform, max_exemplars = self.num_exemplars, scale_factor=32, split="train")
        self.dataset_val = FSCD147_Dataset(root=valdir, transform=self.val_transform, max_exemplars = self.num_exemplars, scale_factor=32, split="val", now_eval=self.args.eval)
        self.dataset_test = FSCD147_Dataset(root=valdir, transform=self.val_transform, max_exemplars = self.num_exemplars, scale_factor=32, split="test", now_eval=self.args.eval)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=1, num_workers=self.hparams.num_workers, collate_fn=self.collate, worker_init_fn=self.worker_seed_func, generator=self.g)
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, num_workers=self.hparams.num_workers, collate_fn=self.collate, worker_init_fn=self.worker_seed_func, generator=self.g)

class FSCDLVISDataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = FSCD_LVIS_Dataset
        self.collate = custom_collate

        self.train_transform = self.transforms_list['default'] 
        self.val_transform =self.transforms_list['default']

    def setup(self, stage: str):
        traindir = self.hparams.datadir
        valdir = self.hparams.datadir

        self.dataset_train = FSCD_LVIS_Dataset(root=traindir, transform=self.train_transform, max_exemplars = self.num_exemplars, scale_factor=32, split="train")
        self.dataset_val = FSCD_LVIS_Dataset(root=valdir, transform=self.val_transform, max_exemplars = self.num_exemplars, scale_factor=32, split="test", now_eval=self.args.eval)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=1, num_workers=self.hparams.num_workers, collate_fn=self.collate, worker_init_fn=self.worker_seed_func, generator=self.g)
    
class FSCDLVISUnseenDataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = FSCD_LVIS_Dataset
        self.collate = custom_collate

        self.train_transform = self.transforms_list['default'] 
        self.val_transform =self.transforms_list['default']

    def setup(self, stage: str):
        traindir = self.hparams.datadir
        valdir = self.hparams.datadir

        self.dataset_train = FSCD_LVIS_Dataset(root=traindir, transform=self.train_transform, max_exemplars = self.num_exemplars, scale_factor=32, split="train", LVIS_split_unseen=True)
        self.dataset_val = FSCD_LVIS_Dataset(root=valdir, transform=self.val_transform, max_exemplars = self.num_exemplars, scale_factor=32, split="test", now_eval=self.args.eval, LVIS_split_unseen=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=1, num_workers=self.hparams.num_workers, collate_fn=self.collate, worker_init_fn=self.worker_seed_func, generator=self.g)
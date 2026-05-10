from .datamodules import *

def build_datamodule(args):

    dataset_dict = {
        'RPINE': RPINEDataModule,
        'FSCD147': FSCD147DataModule,
        'FSCD_LVIS_seen': FSCDLVISDataModule,
        'FSCD_LVIS_unseen': FSCDLVISUnseenDataModule,
    }

    datamodule = dataset_dict[args.dataset](
        args = args,
        datadir=args.datapath,
        batchsize=args.batch_size,
        num_workers=args.num_workers,
        num_exemplars=args.num_exemplars,
    )

    return datamodule
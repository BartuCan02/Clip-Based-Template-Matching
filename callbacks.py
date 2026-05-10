import os
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


class CustomCheckpoint(ModelCheckpoint):
    """
    Checkpoint load & save
    """
    def __init__(self, args):
        self.dirpath = args.logpath

        if not args.eval and not args.resume and not args.multi_gpu:
            assert not os.path.exists(self.dirpath), f'{self.dirpath} already exists'
        self.filename = 'best_model'
        if args.best_model_count:
            self.monitor = 'val/MAE'
            self.mode = 'min'
        else:
            self.monitor = 'val/AP'
            self.mode = 'max'

        super(CustomCheckpoint, self).__init__(dirpath=self.dirpath,
                                               monitor=self.monitor,
                                               filename=self.filename,
                                               mode=self.mode,
                                               verbose=True,
                                               save_last=True,
                                               every_n_epochs=args.AP_term,
                                               )
        # For evaluation, load best_model-v(k).cpkt where k is the max index
        if args.eval:
            self.modelpath = self.return_best_model_path(self.dirpath, self.filename)
            print('evaluating', self.modelpath)
        # For training, set the filename as best_model.ckpt
        # For resuming training, pytorch_lightning will automatically set the filename as best_model-v(k).ckpt
        else:
            self.modelpath = os.path.join(self.dirpath, self.filename + '.ckpt')
        self.lastmodelpath = os.path.join(self.dirpath, 'last.ckpt') if args.resume else None

    def return_best_model_path(self, dirpath, filename):
        ckpt_files = os.listdir(dirpath)  # list of strings
        vers = [ckpt_file for ckpt_file in ckpt_files if filename in ckpt_file]
        vers.sort()
        best_model = vers[-1] if len(vers) == 1 else vers[-2]
        return os.path.join(self.dirpath, best_model)

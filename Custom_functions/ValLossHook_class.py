import torch
from detectron2.data.build import build_detection_test_loader
from detectron2.engine import HookBase
from detectron2.utils import comm

class ValLossHook(HookBase):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self._loader = iter(build_detection_test_loader(self.cfg, cfg.DATASETS.TEST[0]))
        
    def after_step(self):
        """
            After each step calculates the validation loss and adds it to the train storage
        """
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), "The loss can't be infinite; {}".format(loss_dict)

            loss_dict_reduced = {"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(val_total_loss=losses_reduced, **loss_dict_reduced)
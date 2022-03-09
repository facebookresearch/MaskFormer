import os
from torch.utils.tensorboard import SummaryWriter
from detectron2.utils.events import EventWriter, get_event_storage

class CustomTensorboardXWriter(EventWriter):
    """
    Writes scalars and images based on storage key to train or val tensorboard file.
    """

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the base directory to save the output events. This class creates two subdirs in log_dir
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        
        # separate the writers into a train and a val writer
        train_writer_path = os.path.join(log_dir,"train")
        os.makedirs(train_writer_path, exist_ok=True)
        self._writer_train = SummaryWriter(train_writer_path, **kwargs)
        
        val_writer_path = os.path.join(log_dir,"val")
        os.makedirs(val_writer_path, exist_ok=True)
        self._writer_val = SummaryWriter(val_writer_path, **kwargs)

    def write(self):

        storage = get_event_storage()
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if k.startswith("val_"):
                k = k.replace("val_","")
                self._writer_val.add_scalar(k, v, iter)
            else:
                self._writer_train.add_scalar(k, v, iter)

        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                if k.startswith("val_"):
                    k = k.replace("val_","")
                    self._writer_val.add_image(img_name, img, step_num)
                else:
                    self._writer_train.add_image(img_name, img, step_num)
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

        if len(storage._histograms) >= 1:
            for params in storage._histograms:
                self._writer_train.add_histogram_raw(**params)
            storage.clear_histograms()

    def close(self):
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer_train.close()
            self._writer_val.close()

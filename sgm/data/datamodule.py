import torch
import numpy as np
from typing import Optional
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from sgm.util import get_obj_from_str
from .dataset import Dataset


class DataModule(LightningDataModule):
    """PyTorch Lightning data module for paired DESI/Euclid datasets."""

    def __init__(
        self,
        train: Optional[DictConfig] = None,
        validation: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.train_config = train
        self.val_config = validation
        self.test_config = test
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        if self.train_config is not None:
            self.train_dataset = self._create_dataset(self.train_config.dataset)
        if self.val_config is not None:
            self.val_dataset = self._create_dataset(self.val_config.dataset)
        if self.test_config is not None:
            self.test_dataset = self._create_dataset(self.test_config.dataset)

    def _create_dataset(self, config: DictConfig) -> Dataset:
        config_params = config.params if 'params' in config else config

        # Accept both 'paths' (current) and 'path' (legacy) parameter names.
        if 'paths' in config_params:
            paths = config_params.paths
        elif 'path' in config_params:
            paths = config_params.path
        else:
            raise ValueError("Config must contain 'paths' or 'path'")

        if hasattr(paths, 'to_container'):
            paths = paths.to_container()

        dataset_args = {
            'paths': paths,
            'image_folder': config_params.get('image_folder', 'images'),
            'error_folder': config_params.get('error_folder', 'RMS'),
            'label_folder': config_params.get('label_folder', 'labels'),
            'match_files': config_params.get('match_files', False),
            'use_labels': config_params.get('use_labels', False),
            'sample_ratio': config_params.get('sample_ratio', None),
            'crop': config_params.get('crop', False),
            'normalize': config_params.get('normalize', False),
            'standardize': config_params.get('standardize', False),
            'mean': config_params.get('mean', None),
            'std': config_params.get('std', None),
            'target_h': config_params.get('target_h', 256),
            'target_w': config_params.get('target_w', 256),
            'random_crop': config_params.get('random_crop', False),
            'rotation_angle': config_params.get('rotation_angle', 0),
            'random_rotation': config_params.get('random_rotation', False),
            'pixel_scale': config_params.get('pixel_scale', 0.262),
            'error_type': config_params.get('error_type', 'rms'),
            'del_channel': config_params.get('del_channel', None),
            'reproj_wcs': config_params.get('reproj_wcs', False),
            'reproj_mode': config_params.get('reproj_mode', 'exact'),
            'apply_zp_conv': config_params.get('apply_zp_conv', False),
            'from_zp': config_params.get('from_zp', 30),
            'to_zp': config_params.get('to_zp', 22.5),
            'psf_folder': config_params.get('psf_folder', 'PSF'),
            'desi_img_valid_range': config_params.get('desi_img_valid_range', None),
            'desi_err_valid_range': config_params.get('desi_err_valid_range', None),
            'euclid_img_valid_range': config_params.get('euclid_img_valid_range', None),
            'euclid_err_valid_range': config_params.get('euclid_err_valid_range', None),
        }

        if 'custom_fits_reader' in config_params:
            reader = config_params.custom_fits_reader
            if isinstance(reader, str):
                dataset_args['custom_fits_reader'] = get_obj_from_str(reader)
            else:
                dataset_args['custom_fits_reader'] = reader

        return Dataset(**dataset_args)

    def _get_dataloader(self, dataset, loader_config):
        loader_args = dict(loader_config)
        if dataset.is_multi_paths and dataset.match_files:
            loader_args['collate_fn'] = multimodal_collate_fn
        return torch.utils.data.DataLoader(dataset, **loader_args)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Training dataset not configured")
        return self._get_dataloader(self.train_dataset, self.train_config.loader)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Validation dataset not configured")
        return self._get_dataloader(self.val_dataset, self.val_config.loader)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset not configured")
        return self._get_dataloader(self.test_dataset, self.test_config.loader)


def multimodal_collate_fn(batch):
    """Collate function for multi-source matched datasets."""
    if not batch:
        return {}

    collated = {}
    sample_keys = batch[0].keys()

    for key in sample_keys:
        values = [sample[key] for sample in batch]

        if key == 'filename':
            collated[key] = values
        elif isinstance(values[0], np.ndarray):
            try:
                arr = np.stack(values)
                collated[key] = torch.from_numpy(arr)
            except Exception as e:
                print(f"Error stacking {key}: {e}")
                collated[key] = values
        elif isinstance(values[0], (int, float)):
            collated[key] = torch.tensor(values)
        else:
            collated[key] = values

    return collated

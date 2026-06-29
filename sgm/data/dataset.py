import numpy as np
import os
import glob
from typing import Union, List, Dict, Optional
from .base import BaseDataset


class Dataset(BaseDataset):
    """
    Unified dataset class supporting all use cases:
    1. Single folder
    2. Multiple folders (unmatched)
    3. Multiple folders (matched by filename intersection)
    """

    def __init__(
        self,
        paths: Union[str, List[str], Dict[str, str]],
        image_folder='images',
        error_folder='RMS',
        label_folder='labels',
        match_files=False,
        use_labels=False,
        sample_ratio=None,
        **kwargs
    ):
        """
        Initialize dataset.

        Args:
            paths: data path(s), supporting:
                - single string: single folder
                - list of strings: multiple folders
                - dict: multiple labeled folders; values can be a single path or list of paths
                  e.g.: {'euclid': path1, 'desi': path2}
                  or:   {'euclid': [path1, path2], 'desi': [path3, path4]}
            image_folder: image subfolder name
            error_folder: error map subfolder name (default 'RMS')
            label_folder: label subfolder name
            match_files: whether to match files by filename intersection (multi-folder only)
            use_labels: whether to load labels
            sample_ratio: sampling ratio (None = use all data)
            **kwargs: additional arguments passed to BaseDataset
        """
        super().__init__(**kwargs)

        self.image_folder = image_folder
        self.error_folder = error_folder
        self.label_folder = label_folder
        self.match_files = match_files
        self.use_labels = use_labels
        self.sample_ratio = sample_ratio

        self._setup_paths(paths)
        self._collect_files()
        self._apply_sampling()

    def _setup_paths(self, paths):
        """Set up path structure."""
        # Check if dict (including OmegaConf DictConfig)
        if isinstance(paths, dict) or (hasattr(paths, 'items') and not isinstance(paths, str)):
            self.is_dict_paths = True
            # Each value may be a single path or a list of paths
            self.path_dict = {}
            for k, v in paths.items():
                # Check if value is iterable (but not a string)
                if hasattr(v, '__iter__') and not isinstance(v, str):
                    # Value is a list; convert each element to string
                    self.path_dict[k] = [str(p) for p in v]
                else:
                    # Value is a single path string
                    self.path_dict[k] = [str(v)]

            self.path_keys = list(self.path_dict.keys())
            # path_list is now a nested list: [[euclid_paths], [desi_paths]]
            self.path_list = [self.path_dict[k] for k in self.path_keys]
        # Check if iterable (list, tuple, OmegaConf ListConfig)
        elif hasattr(paths, '__iter__') and not isinstance(paths, str):
            self.is_dict_paths = False
            self.path_list = [[str(p)] for p in paths]
            self.path_keys = [f'data_{i}' for i in range(len(self.path_list))]
            self.path_dict = {k: p for k, p in zip(self.path_keys, self.path_list)}
        else:
            # Single string path
            self.is_dict_paths = False
            self.path_list = [[str(paths)]]
            self.path_keys = ['data_0']
            self.path_dict = {'data_0': [str(paths)]}

        # Total number of paths (flattened)
        total_paths = sum(len(paths) for paths in self.path_list)
        self.is_multi_paths = total_paths > 1

    def _get_image_dir(self, base_path):
        """Get image directory."""
        if "20240604" in base_path:
            return base_path
        return os.path.join(base_path, self.image_folder)

    def _collect_files(self):
        """Collect file list."""
        if self.is_multi_paths and self.match_files:
            self._collect_matched_files()
        else:
            self._collect_simple_files()

    def _collect_matched_files(self):
        """Collect matched files (intersection of filenames across folders)."""
        folder_files = {}

        for key, path_list in self.path_dict.items():
            # Merge files from all paths under this key
            all_basenames = set()
            for path in path_list:
                img_dir = self._get_image_dir(path)
                if not os.path.exists(img_dir):
                    print(f"Warning: path {img_dir} does not exist, skipping")
                    continue

                print(f"Collecting files: {key} - {img_dir}...")
                basenames = set()
                with os.scandir(img_dir) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.endswith('.fits'):
                            basenames.add(entry.name)
                all_basenames.update(basenames)
                print(f"  Found {len(basenames)} files")

            folder_files[key] = all_basenames
            print(f"  {key} total: {len(all_basenames)} files")

        if not folder_files:
            self.file_list = []
            print("No files found")
            return

        print("Computing filename intersection...")
        common_files = set.intersection(*folder_files.values())
        self.file_list = sorted(list(common_files))

        print(f"Found {len(self.file_list)} matched files (from {len(folder_files)} sources)")

    def _collect_simple_files(self):
        """Collect simple file list."""
        self.file_list = []

        for idx, path_list in enumerate(self.path_list):
            for path in path_list:
                img_dir = self._get_image_dir(path)
                if not os.path.exists(img_dir):
                    print(f"Warning: path {img_dir} does not exist, skipping")
                    continue

                print(f"Collecting files: {img_dir}...")
                files = glob.glob(os.path.join(img_dir, "*.fits"))
                # Skip size check for speed; assume all FITS files are valid
                basenames = [os.path.basename(f) for f in files]

                if self.is_multi_paths:
                    self.file_list.extend([(idx, bn) for bn in basenames])
                else:
                    self.file_list = basenames

                print(f"  Found {len(basenames)} files")

        total_path_count = sum(len(paths) for paths in self.path_list)
        print(f"Loaded {len(self.file_list)} files total (from {total_path_count} folders)")

    def _apply_sampling(self):
        """Apply sampling."""
        total_size = len(self.file_list)

        if self.sample_ratio is None or self.sample_ratio >= 1.0:
            self.sampled_indices = list(range(total_size))
        else:
            sample_count = int(total_size * self.sample_ratio)
            self.sampled_indices = sorted(
                np.random.choice(total_size, sample_count, replace=False).tolist()
            )

        if len(self.sampled_indices) < total_size:
            print(f"Sampled {len(self.sampled_indices)}/{total_size} items")

    def __len__(self):
        return len(self.sampled_indices)

    def __getitem__(self, idx):
        """
        Get a data item.

        Pipeline:
        1. Reset crop positions and rotation angle (ensure per-sample independence)
        2. Read and process image (rotate -> crop -> normalize -> standardize)
        3. Read and process error map (rotate -> crop, same angle and position as image)
        """
        self.reset_crop_positions()
        self.reset_rotation_angle()

        real_idx = self.sampled_indices[idx]

        if self.is_multi_paths and self.match_files:
            return self._get_matched_item(real_idx)
        elif self.is_multi_paths:
            return self._get_multi_item(real_idx)
        else:
            return self._get_single_item(real_idx)

    def _get_single_item(self, idx):
        """
        Single-folder mode.

        Note: process image first (generates rotation angle and crop position),
              then process error map (uses the same rotation angle and crop position).
        """
        filename = self.file_list[idx]
        path_list = self.path_list[0]

        # Read from the first available path (usually only one)
        path = path_list[0]

        # 1. Process image: rotate -> crop -> normalize -> standardize
        img_path = os.path.join(self._get_image_dir(path), filename)
        image = self.read_fits(img_path)
        image = self.process_image(image)

        data = {'images': image, 'filename': filename}

        # 2. Process error map: same rotation angle and crop position as image
        error = self._read_error_data(path, filename, 'data_0')
        if error is not None:
            data['error'] = error

        # 3. Load labels
        if self.use_labels:
            label_path = os.path.join(path, self.label_folder, filename)
            if os.path.exists(label_path):
                data['label'] = np.load(label_path)

        return data

    def _get_multi_item(self, idx):
        """
        Multi-folder mode (unmatched).

        Note: process image first, then error map (same augmentation).
        """
        folder_idx, filename = self.file_list[idx]
        path_list = self.path_list[folder_idx]
        key = self.path_keys[folder_idx]

        # Find the path within this source that contains the file
        path = None
        for p in path_list:
            img_path = os.path.join(self._get_image_dir(p), filename)
            if os.path.exists(img_path):
                path = p
                break

        if path is None:
            raise FileNotFoundError(f"File {filename} not found in any path of source {key}")

        # 1. Process image
        img_path = os.path.join(self._get_image_dir(path), filename)
        image = self.read_fits(img_path)
        image = self.process_image(image, data_key=key)

        data = {
            'images': image,
            'filename': filename,
            'source': key
        }

        # 2. Process error map (same augmentation as image)
        error = self._read_error_data(path, filename, key)
        if error is not None:
            data['error'] = error

        # 3. Load labels
        if self.use_labels:
            label_path = os.path.join(path, self.label_folder, filename)
            if os.path.exists(label_path):
                data['label'] = np.load(label_path)

        return data

    def _get_matched_item(self, idx):
        """
        Matched mode (multi-source).

        All sources share the same crop position and rotation angle,
        ensuring DESI/Euclid flux maps and error maps are aligned.
        """
        filename = self.file_list[idx]
        data = {'filename': filename}

        # Use a shared crop key so all sources share the same random crop position
        shared_crop_key = '_matched_shared'

        for key, path_list in self.path_dict.items():
            # Find the path within this source that contains the file
            path = None
            for p in path_list:
                img_path = os.path.join(self._get_image_dir(p), filename)
                if os.path.exists(img_path):
                    path = p
                    break

            if path is None:
                # File not found in this source; skip
                continue

            # 1. Process image (all sources share crop position and rotation angle)
            img_path = os.path.join(self._get_image_dir(path), filename)
            image = self.read_fits(img_path)
            image = self.process_image(image, data_key=shared_crop_key)
            data[f'{key}_img'] = image

            # 2. Process error map (same crop position as image)
            error = self._read_error_data(path, filename, shared_crop_key)
            if error is not None:
                data[f'{key}_error'] = error

            # 3. Read PSF (global quantity, no augmentation)
            psf_path = os.path.join(path, self.psf_folder, filename)
            if os.path.exists(psf_path):
                data[f'{key}_psf'] = self.read_psf(psf_path)

        # 4. Load labels
        if self.use_labels:
            first_path_list = self.path_list[0]
            for p in first_path_list:
                label_path = os.path.join(p, self.label_folder, filename)
                if os.path.exists(label_path):
                    data['label'] = np.load(label_path)
                    break

        # 5. Build pixel validity mask (1, H, W) bool
        # Only built when at least one valid_range is configured
        _any_range = any([
            self.desi_img_valid_range,
            self.desi_err_valid_range,
            self.euclid_img_valid_range,
            self.euclid_err_valid_range,
        ])
        if _any_range:
            desi_img_arr   = data.get('desi_img',    None)
            euclid_img_arr = data.get('euclid_img',  None)
            if desi_img_arr is not None and euclid_img_arr is not None:
                data['pixel_mask'] = self.build_pixel_mask(
                    desi_img    = desi_img_arr,
                    euclid_img  = euclid_img_arr,
                    desi_err    = data.get('desi_error',   None),
                    euclid_err  = data.get('euclid_error', None),
                )   # (1, H, W) bool

        return data

    def _read_error_data(self, base_path, filename, data_key):
        """Read error map data."""
        error_path = os.path.join(base_path, self.error_folder, filename)
        return self.read_error_map(error_path, data_key)

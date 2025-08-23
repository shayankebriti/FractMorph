from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
import SimpleITK as sitk


class ACDCDataset(Dataset):
    """
    PyTorch Dataset for ACDC cardiac MRI (ED/ES frames and their segmentations).

    Parameters
    ----------
    dataroot : str
        Root directory containing `{split}_info.json` and image files under `<dataroot>/<split>/`.
    fineSize : tuple of int, optional
        Desired output size as (depth, height, width). Default is (16, 128, 128).
    split : {'train', 'test', 'valid'}, optional
        Dataset split to load. Default is 'train'.
    """
    def __init__(self, dataroot, fineSize=(16, 128, 128), split='train'):
        self.dataroot = dataroot
        self.split = split
        self.fineSize = fineSize

        # Load frame paths from JSON
        info_path = os.path.join(dataroot, f"{split}_info.json")
        with open(info_path, 'r') as f:
            self.entries = json.load(f)

        # Prepend full paths and verify required keys
        for entry in self.entries:
            frames = entry.get("frames", {})
            required = {"ED", "ES", "ED_gt", "ES_gt"}
            if not required.issubset(frames):
                raise KeyError(f"Missing keys in frames: {entry}")
            base = os.path.join(dataroot, split)
            for key in required:
                frames[key] = os.path.join(base, frames[key])

        self.data_len = len(self.entries)

    def __len__(self):
        """Return the number of samples."""
        return self.data_len

    def __getitem__(self, index):
        """
        Load and preprocess a single sample.

        Returns
        -------
        moving : torch.FloatTensor
            ES frame, shape (depth, height, width), values in [0, 1].
        fixed : torch.FloatTensor
            ED frame, shape (depth, height, width), values in [0, 1].
        movingM : torch.FloatTensor
            ES segmentation, shape (1, depth, height, width).
        fixedM : torch.FloatTensor
            ED segmentation, shape (1, depth, height, width).
        """
        paths = self.entries[index]["frames"]

        # Read images and segmentations
        def load_array(path, dtype=np.float32):
            img = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(img).astype(dtype)
            return arr

        data_ED = load_array(paths['ED'])
        data_ES = load_array(paths['ES'])
        seg_ED = load_array(paths['ED_gt'], dtype=np.int32)
        seg_ES = load_array(paths['ES_gt'], dtype=np.int32)

        # Normalize to [0, 1]
        data_ED = (data_ED - data_ED.min()) / (data_ED.max() - data_ED.min())
        data_ES = (data_ES - data_ES.min()) / (data_ES.max() - data_ES.min())

        # Pad spatial dimensions
        d, h, w = data_ED.shape
        _, th, tw = self.fineSize
        sh, sw = (h - th) // 2, (w - tw) // 2
        data_ED = data_ED[:, sh:sh+th, sw:sw+tw]
        data_ES = data_ES[:, sh:sh+th, sw:sw+tw]
        seg_ED = seg_ED[:, sh:sh+th, sw:sw+tw]
        seg_ES = seg_ES[:, sh:sh+th, sw:sw+tw]

        # Handle depth dimension
        td = self.fineSize[0]
        if d >= td:
            sd = (d - td) // 2
            data_ED = data_ED[sd:sd+td]
            data_ES = data_ES[sd:sd+td]
            seg_ED = seg_ED[sd:sd+td]
            seg_ES = seg_ES[sd:sd+td]
        else:
            pad = (td - d) // 2
            def pad_depth(arr):
                buf = np.zeros((td, th, tw), dtype=arr.dtype)
                buf[pad:pad+d] = arr
                return buf
            data_ED = pad_depth(data_ED)
            data_ES = pad_depth(data_ES)
            seg_ED = pad_depth(seg_ED)
            seg_ES = pad_depth(seg_ES)

        # Convert to tensors
        fixed = torch.from_numpy(data_ED).float()
        moving = torch.from_numpy(data_ES).float()
        fixedM = torch.from_numpy(seg_ED).float().unsqueeze(0)
        movingM = torch.from_numpy(seg_ES).float().unsqueeze(0)

        return moving, fixed, movingM, fixedM

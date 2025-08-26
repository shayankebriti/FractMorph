from torch.utils.data import Dataset
import os
import numpy as np
import torch
import SimpleITK as sitk

class LPBA40Dataset(Dataset):
    def __init__(self, dataroot, split='train'):
        self.split = split
        self.dataroot = dataroot

        label_dir = os.path.join(dataroot, 'label')
        cases_dir = os.path.join(dataroot, 'cases')

        self.fixed_path  = os.path.join(cases_dir, 'fixed.nii.gz')
        self.fixed_label = os.path.join(label_dir, 'S01.delineation.structure.label.nii.gz')

        if split == 'train':
            ids = [f"S{idx:02d}" for idx in range(11, 41)]
            subdir = 'train'
        elif split == 'test':
            ids = [f"S{idx:02d}" for idx in range(2, 11)]
            subdir = 'test'
        else:
            raise ValueError(f"Unknown split '{split}'; must be 'train' or 'test'.")

        self.cases = []
        for sid in ids:
            img = os.path.join(cases_dir, subdir, f"{sid}.delineation.skullstripped.nii.gz")
            lbl = os.path.join(label_dir,  f"{sid}.delineation.structure.label.nii.gz")
            if not os.path.isfile(img):
                raise FileNotFoundError(f"Missing image: {img}")
            if not os.path.isfile(lbl):
                raise FileNotFoundError(f"Missing label: {lbl}")
            self.cases.append({'moving': img, 'movingM': lbl})

        self.data_len = len(self.cases)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        fx_img = sitk.ReadImage(self.fixed_path)
        fx_arr = sitk.GetArrayFromImage(fx_img).astype(np.float32)
        fx_lbl_img = sitk.ReadImage(self.fixed_label)
        fx_lbl = sitk.GetArrayFromImage(fx_lbl_img).astype(np.int64)

        case = self.cases[index]
        mv_img = sitk.ReadImage(case['moving'])
        mv_arr = sitk.GetArrayFromImage(mv_img).astype(np.float32)
        mv_lbl_img = sitk.ReadImage(case['movingM'])
        mv_lbl = sitk.GetArrayFromImage(mv_lbl_img).astype(np.int64)

        fixed   = torch.from_numpy(fx_arr).float()
        moving  = torch.from_numpy(mv_arr).float()
        fixedM  = torch.from_numpy(fx_lbl).float().unsqueeze(0)
        movingM = torch.from_numpy(mv_lbl).float().unsqueeze(0)

        return moving, fixed, movingM, fixedM

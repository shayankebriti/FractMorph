import os
import time
import csv
import threading
import numpy as np
import torch
import psutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import listdir
from os.path import join

from torch.utils.data import DataLoader
from torchinfo import summary
from skimage.metrics import hausdorff_distance

from storage.downloader import download_and_save_pth
from storage.uploader import initialize_nextcloud_client, upload_file_to_nextcloud
from utils.STN import SpatialTransformer, Re_SpatialTransformer
from utils.augmentation import MirrorTransform, SpatialTransform
from utils.utils import AverageMeter, to_categorical, dice
from utils.losses import gradient_loss, ncc_loss
from dataset.ACDC.acdc import ACDCDataset as Dataset
from const import BASE_PATH
from plot.utils import save_plot

#################### Change The Targeted Model Here ####################
from models.FractMorph import Head
# from models.FractMorphLight import Head
########################################################################


class RegistrationModel:
    """
    Registration model trainer.
    Supports supervised and unsupervised losses, mixed precision, and checkpointing.
    """
    def __init__(
        self,
        k=0,
        n_channels=1,
        n_classes=8,
        lr=1e-4,
        epochs=1,
        iterations=200,
        batch_size=1,
        model_name='ACDC',
        use_smooth_loss=True,
        use_ncc_loss=True,
        use_fp16=False
    ):
        # Config
        self.k = k
        self.n_classes = n_classes
        self.epochs = epochs
        self.iters = iterations
        self.use_smooth_loss = use_smooth_loss
        self.use_ncc_loss = use_ncc_loss
        self.use_fp16 = use_fp16
        self.model_name = model_name

        # Directories
        self.results_dir    = './artifacts/results'
        self.checkpoint_dir = './artifacts/weights'
        self.plots_dir      = './artifacts/plots'
        os.makedirs(self.plots_dir, exist_ok=True)

        # Data augmentation
        self.mirror_aug = MirrorTransform()
        self.spatial_aug = SpatialTransform(
            do_rotation=True,
            angle_x=(-np.pi/36, np.pi/36),
            angle_y=(-np.pi/36, np.pi/36),
            angle_z=(-np.pi/36, np.pi/36),
            do_scale=True,
            scale=(0.9, 1.1)
        )

        # Model and optimizer
        self.Reger = Head(n_channels=n_channels)
        if torch.cuda.is_available():
            self.Reger = self.Reger.cuda()
        self.optR = torch.optim.Adam(self.Reger.parameters(), lr=lr)

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if (use_fp16 and torch.cuda.is_available()) else None

        # Spatial transformers
        self.stn  = SpatialTransformer()
        self.rstn = Re_SpatialTransformer()
        self.softmax = torch.nn.Softmax(dim=1)

        # Data loaders
        labeled_ds   = Dataset(BASE_PATH, split='train')
        unlabeled_ds = Dataset(BASE_PATH, split='train')
        eval_ds      = Dataset(BASE_PATH, split='eval')
        test_ds      = Dataset(BASE_PATH, split='test')

        self.train_labeled_loader   = DataLoader(labeled_ds,   batch_size=batch_size, shuffle=True)
        self.train_unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True)
        self.eval_loader            = DataLoader(eval_ds,      batch_size=batch_size, shuffle=False)
        self.test_loader            = DataLoader(test_ds,      batch_size=batch_size, shuffle=False)

        # Loss functions and meters
        self.L_smooth     = gradient_loss if use_smooth_loss else None
        self.L_ncc        = ncc_loss      if use_ncc_loss    else None
        self.L_smooth_log = AverageMeter('L_smooth') if use_smooth_loss else None
        self.L_ncc_log    = AverageMeter('L_ncc')    if use_ncc_loss    else None

        # Tracking
        self.epoch_loss         = []
        self.epoch_ncc_loss     = []
        self.epoch_smooth_loss  = []
        self.loss_csv_path      = os.path.join(self.checkpoint_dir, "training_losses.csv")

        # Save model info
        total_params     = sum(p.numel() for p in self.Reger.parameters())
        trainable_params = sum(p.numel() for p in self.Reger.parameters() if p.requires_grad)
        os.makedirs(self.results_dir, exist_ok=True)
        with open(os.path.join(self.results_dir, f"{self.model_name}_params.txt"), 'w') as f:
            f.write(f"Total Params: {total_params:,}\n")
            f.write(f"Trainable Params: {trainable_params:,}\n")
        print(f"[Model Info] Total: {total_params:,}, Trainable: {trainable_params:,}")

    def ensure_batch_and_channel(self, data: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is shaped [B, C, D, H, W]."""
        if data.dim() == 3:
            return data.unsqueeze(0).unsqueeze(0)
        if data.dim() == 4:
            return data.unsqueeze(0)
        return data

    def train_iterator(self, mov, fix, mov_lab=None, fix_lab=None):
        """Single optimization step."""
        # Forward + loss
        if self.use_fp16:
            with torch.cuda.amp.autocast():
                outputs = self.Reger(mov, fix, mov_lab, fix_lab)
                flow = outputs[-1]

                loss_s   = self.L_smooth(flow) if self.use_smooth_loss else torch.tensor(0.0, device=mov.device)
                loss_ncc = torch.mean(self.L_ncc(outputs[0], fix)) if self.use_ncc_loss else torch.tensor(0.0, device=mov.device)

                if self.use_smooth_loss: self.L_smooth_log.update(loss_s.data, mov.size(0))
                if self.use_ncc_loss:    self.L_ncc_log.update(loss_ncc.data, mov.size(0))

                loss = loss_s + loss_ncc

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optR)
            self.scaler.update()
        else:
            outputs = self.Reger(mov, fix, mov_lab, fix_lab)
            flow = outputs[-1]

            loss_s   = self.L_smooth(flow) if self.use_smooth_loss else torch.tensor(0.0, device=mov.device)
            loss_ncc = torch.mean(self.L_ncc(outputs[0], fix)) if self.use_ncc_loss else torch.tensor(0.0, device=mov.device)

            if self.use_smooth_loss: self.L_smooth_log.update(loss_s.data, mov.size(0))
            if self.use_ncc_loss:    self.L_ncc_log.update(loss_ncc.data, mov.size(0))

            (loss_s + loss_ncc).backward()
            self.optR.step()

        self.Reger.zero_grad()
        self.optR.zero_grad()

    def train_epoch(self, epoch: int, augment: bool = True):
        """Run one epoch of training."""
        self.Reger.train()

        pbar = tqdm(range(self.iters), desc=f'Epoch {epoch+1}/{self.epochs}')
        for _ in pbar:
            # ds = self.train_labeled_loader if np.random.rand() < 0.5 else self.train_unlabeled_loader
            ds = self.train_unlabeled_loader
            mov, fix, mov_lab, fix_lab = next(iter(ds))

            if torch.cuda.is_available():
                mov, fix = mov.cuda(), fix.cuda()
                mov_lab = mov_lab.cuda() if mov_lab is not None else None
                fix_lab = fix_lab.cuda() if fix_lab is not None else None

            mov = self.ensure_batch_and_channel(mov)
            fix = self.ensure_batch_and_channel(fix)

            # Augmentation
            if augment:
                code = self.mirror_aug.rand_code()
                coords = self.spatial_aug.rand_coords(mov.shape[-3:])
                mov = self.spatial_aug.augment_spatial(self.mirror_aug.augment_mirroring(mov, code), coords)
                fix = self.spatial_aug.augment_spatial(self.mirror_aug.augment_mirroring(fix, code), coords)
                if mov_lab is not None:
                    mov_lab = self.spatial_aug.augment_spatial(
                        self.mirror_aug.augment_mirroring(mov_lab, code), coords, mode='nearest'
                    )
                    fix_lab = self.spatial_aug.augment_spatial(
                        self.mirror_aug.augment_mirroring(fix_lab, code), coords, mode='nearest'
                    )

            self.train_iterator(mov, fix, mov_lab, fix_lab)
            desc = f'Epoch {epoch+1}/{self.epochs}'
            if self.L_smooth_log: desc += f' | L_smooth: {self.L_smooth_log.avg:.4f}'
            if self.L_ncc_log:    desc += f' | L_ncc: {self.L_ncc_log.avg:.4f}'
            pbar.set_description(desc)

    def train(self, aug=True, save_every_epoch=None):
        """Full training loop with optional checkpointing."""
        os.makedirs(os.path.dirname(self.loss_csv_path), exist_ok=True)
        with open(self.loss_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Overall Loss", "NCC Loss", "Smooth Loss"])
        
        for epoch in range(self.epoches - self.k):
            if self.L_smooth_log is not None:
                self.L_smooth_log.reset()
            if self.L_ncc_log is not None:
                self.L_ncc_log.reset()

            self.train_epoch(epoch + self.k, is_aug=aug)

            smooth_avg = self.L_smooth_log.avg.cpu().item() if self.use_smooth_loss else 0.0
            ncc_avg = self.L_ncc_log.avg.cpu().item() if self.use_ncc_loss else 0.0
            overall_loss = smooth_avg + ncc_avg

            self.epoch_loss.append(overall_loss)
            self.epoch_smooth_loss.append(smooth_avg)
            self.epoch_ncc_loss.append(ncc_avg)

            print(f"Epoch {epoch + self.k + 1} losses: Overall: {overall_loss:.4f}, NCC: {ncc_avg:.4f}, Smooth: {smooth_avg:.4f}")
            with open(self.loss_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch + self.k + 1, overall_loss, ncc_avg, smooth_avg])
        
            # Optionally save checkpoints every n epochs.
            if save_every_epoch is not None and (epoch + 1) % save_every_epoch == 0:
                self.checkpoint(epoch + 1, self.k)

        self.checkpoint(self.epoches - self.k, self.k)
        self.plot_losses()

    def checkpoint(self, epoch: int):
        """Save model and upload asynchronously."""
        path = f"{self.checkpoint_dir}/Reger_{self.model_name}_epoch_{epoch}.pth"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.Reger.state_dict(), path, _use_new_zipfile_serialization=False)
        print(f"Checkpoint saved: {path}")

        def _upload():
            try:
                nc = initialize_nextcloud_client()
                upload_file_to_nextcloud(nc, folder_name="weights", file_path=path)
            except Exception as e:
                print(f"[Upload] {e}")

        threading.Thread(target=_upload, daemon=True).start()

    def load(self, weight_path: str = None):
        """Load weights (local path or URL)."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if weight_path and weight_path.startswith(('http://', 'https://')):
            weight_path = download_and_save_pth(weight_path, self.checkpoint_dir)
        elif not weight_path:
            weight_path = f"{self.checkpoint_dir}/Reger_{self.model_name}_epoch_{self.k}.pth"

        state = torch.load(weight_path, map_location=device)
        self.Reger.load_state_dict(state)
        self.Reger.to(device)

    def evaluate(self, mode: str = 'test'):
        """
        Save raw flow fields for a given split: 'train', 'eval', or 'test'.
        """
        base = os.path.join(self.results_dir, self.model_name, 'evaluation', mode)
        flows = os.path.join(base, 'flows')
        os.makedirs(flows, exist_ok=True)

        loader = {
            'train': self.train_labeled_loader,
            'eval':  self.eval_loader,
            'test':  self.test_loader
        }.get(mode)
        if loader is None:
            raise ValueError("Mode must be 'train', 'eval', or 'test'.")

        self.Reger.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def ensure5d(x):
            if x.dim() == 3: return x.unsqueeze(0).unsqueeze(0)
            if x.dim() == 4: return x.unsqueeze(0)
            return x

        for i, (mov, fix, mov_lab, fix_lab) in enumerate(loader):
            mov_t  = ensure5d(mov).to(device)
            fix_t  = ensure5d(fix).to(device)
            mov_l  = ensure5d(mov_lab).to(device)
            fix_l  = ensure5d(fix_lab).to(device)
            with torch.no_grad():
                *_, flow = self.Reger(mov_t, fix_t, mov_l, fix_l)
            np.savez_compressed(os.path.join(flows, f"flow_case_{i:03d}.npz"), flow=flow[0].cpu().numpy())

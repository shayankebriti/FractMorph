import os
import random
import subprocess
import platform
import warnings
from os import listdir
from os.path import join

import numpy as np
import psutil
import torch
import torch.nn as nn
from cpuinfo import get_cpu_info
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset.ACDC.intialize import init_acdc
from plot.ACDC import *
from storage.uploader import (
    zip_and_upload_artifacts,
    upload_weights_to_nextcloud,
)
from storage.downloader import download_and_extract_zip_from_nextcloud
from train import RegistrationModel

warnings.filterwarnings('ignore')


def print_nvidia_smi():
    print("=== GPU Info (nvidia-smi) ===")
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error executing nvidia-smi:", e)
    except FileNotFoundError:
        print("nvidia-smi not found; ensure NVIDIA drivers are installed.")


def print_cpu_info():
    print("\n=== CPU Info ===")
    print(f"Processor: {platform.processor()}")
    print(f"Machine:   {platform.machine()}")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical cores:  {psutil.cpu_count(logical=True)}")
    print(f"Total CPU usage: {psutil.cpu_percent(interval=1)}%")

    info = get_cpu_info() or {}
    model = info.get('brand_raw') or info.get('hz_advertised_friendly')
    if model:
        print(f"CPU Model: {model}")
    elif platform.system() == "Linux":
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        print("CPU Model:", line.split(":", 1)[1].strip())
                        break
        except Exception as e:
            print("Could not read /proc/cpuinfo:", e)


def print_ram_info():
    print("\n=== RAM Info ===")
    vm = psutil.virtual_memory()
    print(f"Total:     {vm.total / (1024**3):.2f} GB")
    print(f"Available: {vm.available / (1024**3):.2f} GB")
    print(f"Used:      {vm.used / (1024**3):.2f} GB  ({vm.percent}%)")

    print("\n=== RAM Modules ===")
    if platform.system() == "Windows":
        cmd = ["wmic", "memorychip", "get", "BankLabel,Capacity,Speed,Manufacturer,PartNumber"]
    else:
        cmd = ["sudo", "dmidecode", "--type", "17"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except Exception as e:
        print("Could not get RAM module info:", e)


def print_disk_info():
    print("\n=== Disk Info ===")
    for p in psutil.disk_partitions():
        usage = psutil.disk_usage(p.mountpoint)
        print(f"{p.device} ({p.fstype}) mounted on {p.mountpoint}:")
        print(
            f"  Total: {usage.total / (1024**3):.2f} GB  "
            f"Used: {usage.used / (1024**3):.2f} GB  "
            f"Free: {usage.free / (1024**3):.2f} GB  ({usage.percent}%)"
        )

    print("\n=== Physical Disk Models ===")
    if platform.system() == "Linux":
        cmd = ["lsblk", "-d", "-o", "NAME,MODEL,SIZE"]
    else:
        cmd = ["wmic", "diskdrive", "get", "Model,Size"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except Exception as e:
        print("Could not get disk model info:", e)


def seed_everything(seed: int = 42):
    """Seed random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize():
    """Run initialization routines."""
    # print_nvidia_smi()
    # print_cpu_info()
    # print_ram_info()
    # print_disk_info()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"Torch Version: {torch.__version__}")
    seed_everything(0)
    init_acdc()


if __name__ == '__main__':
    # ---- INIT ----
    initialize()
    
    # ---- MODEL ----
    model_name = 'FractMorph'
    print(f"[Model Info] Model Name: {model_name}")
    RSTNet = RegistrationModel(epoches=400, lr=1e-4, use_ncc_loss=True, use_smooth_loss=True, model_name=model_name, use_fp16=False)

    # ---- TRAIN ----
    # RSTNet.load(weight_path="")
    RSTNet.train(save_every_epoch=100)
    upload_weights_to_nextcloud(folder_name=f"weights/{model_name}")

    # ---- EVAL ----
    # RSTNet.load(weight_path="")
    # RSTNet.evaluate(mode='eval')
    # RSTNet.evaluate(mode='test')

    # ---- ARTIFACTS ----
    zip_and_upload_artifacts(folder_name=model_name, in_separate_zips=True)

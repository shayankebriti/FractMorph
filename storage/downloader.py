from const import DATASET_DRIVE_ID, DATASET_STORAGE_URL
import os
import zipfile

import gdown
from nextcloud_client import Client


def download_and_extract_drive(file_id: str = DATASET_DRIVE_ID, output_dir: str = '.') -> None:
    """
    Download a ZIP archive from Google Drive by ID and extract it.

    Parameters
    ----------
    file_id : str
        Google Drive file ID for the ZIP.
    output_dir : str
        Directory in which to save and extract the archive.
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"{file_id}.zip")

    if not os.path.exists(zip_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    print(f"Extracted contents to '{output_dir}'")


def download_and_extract_nextcloud(public_link: str = DATASET_STORAGE_URL,
                                   zip_name: str = 'ACDC.zip',
                                   output_dir: str = '.') -> None:
    """
    Download a ZIP archive from Nextcloud public link and extract it.

    Parameters
    ----------
    public_link : str
        Nextcloud public link to the ZIP.
    zip_name : str
        Name to use when saving the downloaded ZIP.
    output_dir : str
        Directory in which to save and extract the archive.
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, zip_name)

    client = Client.from_public_link(public_link)
    client.get_file('', zip_path)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Failed to download '{zip_name}'")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    print(f"Extracted contents to '{output_dir}'")


def download_and_extract_zip() -> None:
    """Wrapper to download and extract the dataset via Nextcloud."""
    print("Starting dataset download...")
    download_and_extract_nextcloud()


def download_model_weights(url: str, save_dir: str) -> str:
    """
    Download a .pth model weights file from Google Drive or Nextcloud.

    Parameters
    ----------
    url : str
        URL to the .pth file (Google Drive or Nextcloud).
    save_dir : str
        Directory in which to save the weights.

    Returns
    -------
    str
        Full path to the downloaded .pth file.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "loading_weight.pth")

    if os.path.exists(save_path):
        print(f"Weights already exist at '{save_path}'")
        return save_path

    if "drive.google.com" in url:
        print(f"Downloading weights from Google Drive: {url}")
        gdown.download(url, save_path, quiet=False)
    else:
        print(f"Downloading weights from Nextcloud: {url}")
        Client.from_public_link(url).get_file('', save_path)

    return save_path

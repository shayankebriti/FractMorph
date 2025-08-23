import shutil
from pathlib import Path

import nextcloud_client
from const import STORAGE_URL, STORAGE_USERNAME, STORAGE_PASSWORD


def initialize_nextcloud_client() -> nextcloud_client.Client:
    """
    Initialize and authenticate a Nextcloud client.

    Returns
    -------
    nextcloud_client.Client
        Authenticated Nextcloud client.
    """
    nc = nextcloud_client.Client(STORAGE_URL)
    nc.login(STORAGE_USERNAME, STORAGE_PASSWORD)
    return nc


def zip_artifacts_folder(artifacts_dir: Path, zip_name: str = 'artifacts.zip') -> Path:
    """
    Create a ZIP archive of the given artifacts directory, excluding the archive itself.

    Parameters
    ----------
    artifacts_dir : Path
        Directory containing artifacts to zip.
    zip_name : str, optional
        Name of the ZIP file to create.

    Returns
    -------
    Path
        Path to the created ZIP file.
    """
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {artifacts_dir}")

    zip_path = artifacts_dir / zip_name
    base_name = zip_path.with_suffix('')  # Removes .zip for make_archive

    # Exclude existing archive from being re-zipped
    temp_dir = artifacts_dir / '.zip_temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    for item in artifacts_dir.iterdir():
        if item.name != zip_name:
            target = temp_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    shutil.make_archive(str(base_name), 'zip', root_dir=temp_dir)
    shutil.rmtree(temp_dir)

    return zip_path


def upload_file(nc: nextcloud_client.Client, remote_folder: str, local_path: Path) -> None:
    """
    Upload a single file to a specified Nextcloud folder (replacing it if it exists).

    Parameters
    ----------
    nc : nextcloud_client.Client
        Authenticated Nextcloud client.
    remote_folder : str
        Name of the folder in Nextcloud.
    local_path : Path
        Path to the local file to upload.
    """
    # Ensure folder exists (delete if present, then recreate)
    try:
        nc.delete(remote_folder)
    except Exception:
        pass

    try:
        nc.mkdir(remote_folder)
    except Exception:
        pass

    remote_path = f"{remote_folder}/{local_path.name}"
    nc.put_file(remote_path, str(local_path))


def upload_folder_contents(nc: nextcloud_client.Client, remote_folder: str, local_dir: Path) -> None:
    """
    Upload all files from a local directory into a Nextcloud folder.

    Parameters
    ----------
    nc : nextcloud_client.Client
        Authenticated Nextcloud client.
    remote_folder : str
        Target folder name in Nextcloud.
    local_dir : Path
        Local directory whose files will be uploaded.
    """
    # Prepare remote folder
    try:
        nc.delete(remote_folder)
    except Exception:
        pass
    try:
        nc.mkdir(remote_folder)
    except Exception:
        pass

    # Upload each file
    for file_path in sorted(local_dir.iterdir()):
        if file_path.is_file():
            upload_file(nc, remote_folder, file_path)


def zip_and_upload_artifacts(remote_folder: str,
                             artifacts_dir: Path = Path('./artifacts'),
                             separate: bool = False) -> None:
    """
    Zip and upload artifacts to Nextcloud.

    Parameters
    ----------
    remote_folder : str
        Nextcloud folder name where archives will be placed.
    artifacts_dir : Path, optional
        Local artifacts directory. Defaults to './artifacts'.
    separate : bool, optional
        If True, create and upload separate ZIPs for each subfolder (excluding 'weights').
        If False, zip the entire artifacts_dir into one archive.
    """
    nc = initialize_nextcloud_client()

    if separate:
        if not artifacts_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {artifacts_dir}")
        for subdir in artifacts_dir.iterdir():
            if subdir.is_dir() and subdir.name != 'weights':
                zip_path = zip_artifacts_folder(subdir, zip_name=f"{subdir.name}.zip")
                upload_file(nc, remote_folder, zip_path)
    else:
        zip_path = zip_artifacts_folder(artifacts_dir)
        upload_file(nc, remote_folder, zip_path)

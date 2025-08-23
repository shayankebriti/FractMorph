from pathlib import Path

"""
Configuration constants for ACDC dataset paths and storage access.
"""

# Dataset directories
BASE_PATH = Path("ACDC")
TRAIN_PATH = BASE_PATH / "train"
VALID_PATH = BASE_PATH / "valid"
TEST_PATH  = BASE_PATH / "test"

# Storage endpoints
STORAGE_URL          = ""
DATASET_DRIVE_ID     = ""
DATASET_STORAGE_URL  = ""

# Storage credentials
STORAGE_USERNAME = ""
STORAGE_PASSWORD = ""

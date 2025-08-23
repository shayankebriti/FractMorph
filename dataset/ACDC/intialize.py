from dataset.ACDC.acdc import *
from storage.downloader import download_and_extract_zip
from dataset.ACDC.preprocess import preprocess_acdc_directory
from const import *

def init_acdc():
    download_and_extract_zip()
    preprocess_acdc_directory(TRAIN_PATH, BASE_PATH + "/train_info.json")
    preprocess_acdc_directory(TEST_PATH, BASE_PATH + "/valid_info.json")
    preprocess_acdc_directory(TEST_PATH, BASE_PATH + "/test_info.json")
import os
import requests
import zipfile
import tarfile
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Function to download assets
def models_download(model_name="balanceds_k5fold_convnext_small"):
    models_dir = kagglehub.model_download(f"gedewahyupurnama/convnext-multi-view/tensorFlow2/{model_name}")
    return models_dir

def download_example_dicom():
    api.dataset_download_file('gedewahyupurnama/multi-view-dataset-v2', 'example_dicom_tar_gz', quiet=False)
    with zipfile.ZipFile('example_dicom_tar_gz.zip', 'r') as zip_ref:
        with zip_ref.open('example_dicom_tar_gz', 'r') as tarr:
            with tarfile.open(fileobj=tarr, mode='r:gz') as tar_ref:
                tar_ref.extractall()
    print("example dicom files downloaded")





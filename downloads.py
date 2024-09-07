import os
import requests
import zipfile
import tarfile
import subprocess
import kagglehub

print("kagglehub v"+kagglehub.__version__)

# Function to download assets
def models_download(model_name="balanceds_k5fold_convnext_small"):
    models_dir = kagglehub.model_download(f"gedewahyupurnama/convnext-multi-view/tensorFlow2/{model_name}")
    return models_dir

def download_example_dicom():
    path = kagglehub.dataset_download("gedewahyupurnama/multi-view-dataset-v2", "example_dicom_tar_gz")
    with zipfile.ZipFile(path, 'r') as zip_ref:
        with zip_ref.open('example_dicom_tar_gz', 'r') as tarr:
            with tarfile.open(fileobj=tarr, mode='r:gz') as tar_ref:
                tar_ref.extractall()
    print("example dicom files downloaded")
    return path

import tarfile
import kagglehub

print("kagglehub v"+kagglehub.__version__)

# Function to download assets
def models_download():
    models_dir = kagglehub.model_download("gedewahyupurnama/MammographyMultiView/tensorFlow2/mmv_model_4folds")
    return models_dir

def download_example_dicom():
    path = kagglehub.dataset_download("gedewahyupurnama/multi-view-dataset-v2", "example_dicom_tar_gz")
    with tarfile.open(path, mode='r:gz') as tar_ref:
        tar_ref.extractall()
    print("example dicom files downloaded")
    return path

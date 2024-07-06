import requests
import os
import zipfile
import tarfile
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()
repo_owner = "yunama47"
repo_name = "Klasifikasi-Mammografi"
token = os.getenv('GITHUB_TOKEN')
tag_name = "v1.1"

api = KaggleApi()
api.authenticate()

def get_release_assets(repo_owner, repo_name, token, tag_name):
    # GitHub API endpoint to list releases
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"

    # Headers for authentication
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Get the list of releases
    response = requests.get(url, headers=headers)
    releases = response.json()

    if response.status_code != 200:
        print("Failed to fetch releases")
        print(releases)
        return None

    # Find the release by tag name
    release = next((release for release in releases if release["tag_name"] == tag_name), None)

    if not release:
        print(f"No release found with tag name: {tag_name}")
        return None

    # Get the list of assets for the release
    assets_url = release["assets_url"]
    response = requests.get(assets_url, headers=headers)
    assets = response.json()

    if response.status_code != 200:
        print("Failed to fetch assets")
        print(assets)
        return None

    # Extract asset names and download URLs
    asset_list = [(asset["name"], asset["browser_download_url"]) for asset in assets]

    return asset_list


# Function to download assets
def models_download(download_folder, repo_owner=repo_owner, repo_name=repo_name, token=token, tag_name=tag_name):
    asset_list = get_release_assets(repo_owner, repo_name, token, tag_name)
    os.makedirs(download_folder, exist_ok=True)
    for asset_name, download_url in asset_list:
        download_path = os.path.join(download_folder, asset_name)
        if not os.path.exists(download_path):
            print(f"Downloading {asset_name} from {download_url}")
            response = requests.get(download_url, headers={"Authorization": f"token {token}"})

            if response.status_code == 200:
                with open(download_path, "wb") as f:
                    f.write(response.content)
                print(f"Successfully downloaded {asset_name}")
            else:
                print(f"Failed to download {asset_name}")
                print(response.json())
        else:
            print(f"Already downloaded {asset_name} in {download_path}")

def download_example_dicom():
    api.dataset_download_file('gedewahyupurnama/multi-view-dataset-v2', 'example_dicom_tar_gz', quiet=False)
    with zipfile.ZipFile('example_dicom_tar_gz.zip', 'r') as zip_ref:
        with zip_ref.open('example_dicom_tar_gz', 'r') as tarr:
            with tarfile.open(fileobj=tarr, mode='r:gz') as tar_ref:
                tar_ref.extractall()
    print("example dicom files downloaded")




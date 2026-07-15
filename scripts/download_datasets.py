import os
import urllib.request
import zipfile
import ssl
from pathlib import Path

def download_and_extract(url: str, extract_to: Path, dataset_name: str):
    """Downloads a zip file from a URL and extracts it to the target directory."""
    print(f"--- Downloading {dataset_name} ---")
    print(f"URL: {url}")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    zip_path = extract_to / f"{dataset_name}.zip"
    
    # Disable SSL verification for older academic servers if needed
    context = ssl._create_unverified_context()
    
    try:
        # Download with a basic progress message
        print(f"Downloading (this may take a few minutes)...")
        urllib.request.urlretrieve(url, zip_path, context=context)
        print(f"Download complete. Extracting...")
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
        print(f"Extracted {dataset_name} successfully.")
        
        # Cleanup
        os.remove(zip_path)
        print(f"Cleaned up temporary zip file.\n")
        
    except Exception as e:
        print(f"❌ Failed to download or extract {dataset_name}: {e}\n")

if __name__ == "__main__":
    # Base data directory
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    
    # EuroSAT (Zenodo)
    EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
    EUROSAT_DIR = DATA_DIR / "eurosat"
    
    # UC Merced
    UCMERCED_URL = "http://vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"
    UCMERCED_DIR = DATA_DIR / "uc_merced"
    
    print("Starting automated dataset downloads...\n")
    
    download_and_extract(EUROSAT_URL, EUROSAT_DIR, "EuroSAT_RGB")
    download_and_extract(UCMERCED_URL, UCMERCED_DIR, "UCMerced_LandUse")
    
    print("Download script finished.")
    print("Note: You may need to move the extracted folders around slightly to match the exact `data/eurosat/Class` structure depending on how the zips were packed.")

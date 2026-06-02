import os
import shutil
import zipfile
import sys
import json

def setup_kaggle_credentials():
    """Helps the user set up their Kaggle credentials on Windows."""
    user_home = os.path.expanduser("~")
    kaggle_dir = os.path.join(user_home, ".kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json_path):
        print(f"[✓] Kaggle credentials found at {kaggle_json_path}")
        return True
        
    print("\n" + "="*60)
    print(" KAGGLE CREDENTIALS SETUP ")
    print("="*60)
    print("To download datasets from Kaggle, you need an API token:")
    print("1. Go to https://www.kaggle.com (Log in or sign up)")
    print("2. Navigate to your Account Settings (click your profile photo -> Settings)")
    print("3. Scroll down to the 'API' section and click 'Create New Token'")
    print("4. This will download a file named 'kaggle.json'")
    print("="*60 + "\n")
    
    # Check if user has downloaded it to their Downloads folder
    downloads_dir = os.path.join(user_home, "Downloads")
    downloaded_json = os.path.join(downloads_dir, "kaggle.json")
    local_json = "kaggle.json"
    
    source_json = None
    if os.path.exists(local_json):
        source_json = local_json
    elif os.path.exists(downloaded_json):
        source_json = downloaded_json
        
    if source_json:
        print(f"[!] Found 'kaggle.json' in {os.path.dirname(source_json)}.")
        confirm = input("Would you like to automatically configure it? (y/n): ").strip().lower()
        if confirm == 'y':
            os.makedirs(kaggle_dir, exist_ok=True)
            shutil.copy(source_json, kaggle_json_path)
            # Set permissions (on Windows this is less strict but good practice)
            try:
                os.chmod(kaggle_json_path, 0o600)
            except Exception:
                pass
            print(f"[✓] Successfully configured kaggle.json at {kaggle_json_path}")
            return True
            
    print("[x] kaggle.json not found in Downloads or the current folder.")
    print(f"Please copy the downloaded 'kaggle.json' file to: {kaggle_json_path}")
    input("Press Enter once you have copied the file to continue...")
    
    if os.path.exists(kaggle_json_path):
        print("[✓] Kaggle credentials detected!")
        return True
    else:
        print("[x] Credentials still missing. Cannot automate Kaggle download.")
        return False

def download_and_extract():
    if not setup_kaggle_credentials():
        print("Please configure Kaggle API credentials to proceed.")
        return
        
    # Ensure kaggle package is installed
    try:
        import kaggle
    except ImportError:
        print("[!] Installing the Kaggle library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        import kaggle
        
    dataset_slug = "xhlulu/140k-real-and-fake-faces-dataset"
    download_dir = "./dataset_temp"
    
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"\n[~] Downloading dataset '{dataset_slug}'...")
    print("This dataset is large (~1.2 GB). Please ensure you have a stable connection.")
    
    try:
        # download the dataset
        kaggle.api.dataset_download_files(dataset_slug, path=download_dir, unzip=True)
        print("[✓] Download completed!")
    except Exception as e:
        print(f"[x] Error downloading dataset: {e}")
        return

    # Now, let's restructure it to match the notebook structure
    # Expected: Dataset/Train/Real, Dataset/Train/Fake, Dataset/Validation/Real...
    target_base = "./Dataset"
    
    # xhlulu structure:
    # dataset_temp/
    #   train/
    #     real/
    #     fake/
    #   validation/
    #     real/
    #     fake/
    #   test/
    #     real/
    #     fake/
    
    print("\n[~] Restructuring files to match the expected Colab structure...")
    
    source_base = os.path.join(download_dir, "real_and_fake")
    if not os.path.exists(source_base):
        # Sometimes it unzips directly in download_dir
        source_base = download_dir
        
    mapping = {
        "train": "Train",
        "validation": "Validation",
        "test": "Test"
    }
    
    class_mapping = {
        "real": "Real",
        "fake": "Fake"
    }
    
    for src_split, target_split in mapping.items():
        src_split_path = os.path.join(source_base, src_split)
        if not os.path.exists(src_split_path):
            # Check if folders are capitalized in source
            src_split_path = os.path.join(source_base, target_split)
            if not os.path.exists(src_split_path):
                continue
                
        for src_class, target_class in class_mapping.items():
            src_class_path = os.path.join(src_split_path, src_class)
            if not os.path.exists(src_class_path):
                src_class_path = os.path.join(src_split_path, target_class)
                if not os.path.exists(src_class_path):
                    continue
                    
            target_path = os.path.join(target_base, target_split, target_class)
            os.makedirs(target_path, exist_ok=True)
            
            print(f"Moving files for {target_split}/{target_class}...")
            # Move all files
            for file_name in os.listdir(src_class_path):
                shutil.move(
                    os.path.join(src_class_path, file_name),
                    os.path.join(target_path, file_name)
                )
                
    # Cleanup temp directory
    try:
        shutil.rmtree(download_dir)
        print("[✓] Restructuring complete! Dataset is located at: ./Dataset")
    except Exception as e:
        print(f"[!] Warning: Could not clean up temporary download folder: {e}")
        
    print("\n" + "="*60)
    print(" GOOGLE COLAB / GOOGLE DRIVE PREPARATION ")
    print("="*60)
    print("To use this dataset in your Google Colab notebook:")
    print("1. Compress this new 'Dataset' folder into a ZIP file named 'Dataset.zip'.")
    print("2. Upload 'Dataset.zip' to your Google Drive in the main folder (My Drive).")
    print("3. In the Colab notebook, set:")
    print("   zip_path = \"/content/drive/MyDrive/Dataset.zip\"")
    print("4. Run the notebook to train your model on 140,000 images!")
    print("="*60 + "\n")

if __name__ == "__main__":
    download_and_extract()

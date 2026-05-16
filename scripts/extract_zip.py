import zipfile
import os
import argparse
import shutil
import tempfile

def process_spectral_images(zip_path, final_destination):
    """
    Extracts zip to a temp dir, finds SPECTRAL_IMAGE.TIF files, 
    renames them to parent folder name, and moves them to final_destination.
    """
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} does not exist.")
        return

    if not os.path.exists(final_destination):
        os.makedirs(final_destination)

    # Use a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting {zip_path} to temporary directory...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is not a valid zip file.")
            return
        except Exception as e:
            print(f"An unexpected error occurred during extraction: {e}")
            return

        print("Searching for SPECTRAL_IMAGE.TIF files and renaming...")
        found_any = False
        for root, dirs, files in os.walk(temp_dir):
            if "SPECTRAL_IMAGE.TIF" in files:
                found_any = True
                source_path = os.path.join(root, "SPECTRAL_IMAGE.TIF")
                parent_name = os.path.basename(root)
                new_filename = f"{parent_name}.TIF"
                target_path = os.path.join(final_destination, new_filename)
                
                print(f"Moving and renaming: {source_path} -> {target_path}")
                shutil.move(source_path, target_path)

        if not found_any:
            print("No SPECTRAL_IMAGE.TIF files found in the zip.")
        else:
            print(f"Processing complete. Kept only SPECTRAL_IMAGE.TIF files in {final_destination}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SPECTRAL_IMAGE.TIF files from a zip and rename them.")
    parser.add_argument(
        "zip_path", 
        help="Path to the zip file to extract"
    )

    args = parser.parse_args()

    # Destination is fixed to 'data' as per requirements
    process_spectral_images(args.zip_path, "data")

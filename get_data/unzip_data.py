import zipfile
import os

def unzip_folder(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_to)
    print(f"Zip file '{zip_path}' extracted successfully to '{extract_to}'.")


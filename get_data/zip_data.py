import os 
import zipfile

def zip_folder(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_LZMA) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, start=folder_path)
                zipf.write(full_path, arcname)
    print(f"Folder '{folder_path}' zipped successfully to '{output_zip_path}'.")


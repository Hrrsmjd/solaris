import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def upload_h5_files(local_folder, repo_id, files_to_upload=None):
    api = HfApi()

    # Get list of files already in the repository
    existing_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    if files_to_upload is None:
        # If no specific files are provided, get all .h5 files in the local folder
        files_to_upload = [f for f in os.listdir(local_folder) if f.endswith(".h5")]

    # Upload each file that doesn't already exist
    for file in files_to_upload:
        if file.endswith(".h5") and os.path.exists(os.path.join(local_folder, file)):
            if file not in existing_files:
                local_path = os.path.join(local_folder, file)
                remote_path = file  # Keep the same filename on Hugging Face

                print(f"Uploading {file}...")
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"Uploaded {file} successfully.")
            else:
                print(f"Skipping {file} as it already exists in the repository.")
        else:
            print(f"Skipping {file} as it's not a .h5 file or doesn't exist in the local folder.")


def delete_npz_files(repo_id):
    api = HfApi()

    # Get list of files in the repository
    existing_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    # Filter .npz files
    npz_files = [f for f in existing_files if f.endswith(".npz")]

    # Delete each .npz file
    for file in npz_files:
        print(f"Deleting {file}...")
        api.delete_file(path_in_repo=file, repo_id=repo_id, repo_type="dataset")
        print(f"Deleted {file} successfully.")

def main():
    parser = argparse.ArgumentParser(description="Upload or clean files in the Solaris Hugging Face dataset repo.")
    parser.add_argument("--repo-id", default="hrrsmjd/AIA_12hour_512x512")
    parser.add_argument("--local-folder", type=Path, default=None)
    parser.add_argument("--upload", nargs="*", default=None, help="Optional list of .h5 files to upload.")
    parser.add_argument(
        "--delete-npz",
        action="store_true",
        help="Delete all .npz files from the dataset repository.",
    )
    args = parser.parse_args()

    if args.upload is not None:
        if args.local_folder is None:
            raise ValueError("--local-folder is required when uploading files.")
        upload_h5_files(str(args.local_folder), args.repo_id, args.upload or None)

    if args.delete_npz:
        delete_npz_files(args.repo_id)

    if args.upload is None and not args.delete_npz:
        parser.error("Specify at least one action: --upload or --delete-npz.")


if __name__ == "__main__":
    main()

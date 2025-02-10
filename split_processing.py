#  debugging of file was assisted with AI
import os
import shutil
import random

def create_splits(root_dir,
                    test_list_filename="List_of_testing_videos.txt",
                    subfolders=["Celeb-synthesis", "Youtube-real", "Celeb-real"],
                    train_ratio=0.8):
    # Create output directories
    test_dir = os.path.join(root_dir, "test")
    train_dir = os.path.join(root_dir, "train")
    val_dir   = os.path.join(root_dir, "val")
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Read the test file and parse each line
    test_list_path = os.path.join(root_dir, test_list_filename)
    test_paths = set()
    with open(test_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rel_path = parts[1].strip()
                test_paths.add(rel_path.lower())

    remaining_files = [] 

    # Loop through each specified subfolder
    for folder in subfolders:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"Folder '{folder_path}' not found. Skipping.")
            continue

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".mp4"):
                # Construct the relative path as it would appear in the test list.
                relative_path = os.path.join(folder, file_name).replace("\\", "/")
                # Compare in lower-case just in case
                if relative_path.lower() in test_paths:
                    dest_file = os.path.join(test_dir, file_name)
                    print(f"Copying {os.path.join(folder_path, file_name)} to {dest_file} (test set)")
                    shutil.copy2(os.path.join(folder_path, file_name), dest_file)
                else:
                    remaining_files.append(os.path.join(folder_path, file_name))

    # Shuffle the remaining (non-test) videos
    random.shuffle(remaining_files)
    num_remaining = len(remaining_files)
    num_train = int(train_ratio * num_remaining)

    train_files = remaining_files[:num_train]
    val_files = remaining_files[num_train:]

    # Copy training files
    for source_file in train_files:
        file_name = os.path.basename(source_file)
        dest_file = os.path.join(train_dir, file_name)
        print(f"Copying {source_file} to {dest_file} (train set)")
        shutil.copy2(source_file, dest_file)

    # Copy validation files
    for source_file in val_files:
        file_name = os.path.basename(source_file)
        dest_file = os.path.join(val_dir, file_name)
        print(f"Copying {source_file} to {dest_file} (val set)")
        shutil.copy2(source_file, dest_file)

    print(f"Done. Total non-test videos: {num_remaining}. Train: {len(train_files)}, Val: {len(val_files)}.")

if __name__ == "__main__":

    # set root directory
    root_directory = "Celeb-DF"

    create_splits(root_directory)


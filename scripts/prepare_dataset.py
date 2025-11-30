#!/usr/bin/env python3
"""
Dataset preparation script for Stable Diffusion fine-tuning
Works with partial datasets and handles COCO data
"""

import os
import json
import zipfile
from PIL import Image
import logging
from pathlib import Path
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_partial_data():
    """
    Check what data is available for training
    """
    print("[DATA] Checking available datasets...")

    data_sources = []

    # Check for partial zip files
    for file in os.listdir('.'):
        if file.endswith('.zip.part') or file.endswith('.zip'):
            file_path = Path(file)
            size_mb = file_path.stat().st_size / (1024 * 1024)

            print(f"[FOUND] {file}: {size_mb:.1f} MB")

            if file.endswith('.zip'):
                data_sources.append({
                    'path': file,
                    'type': 'zip',
                    'size_mb': size_mb,
                    'status': 'complete'
                })
            else:
                data_sources.append({
                    'path': file,
                    'type': 'partial_zip',
                    'size_mb': size_mb,
                    'status': 'downloading'
                })

    # Check for extracted directories
    potential_dirs = ['train2014', 'val2014', 'dataset', 'data', 'images']
    for dir_name in potential_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            image_count = len(list(Path(dir_name).rglob('*.jpg'))) + len(list(Path(dir_name).rglob('*.png')))
            print(f"[FOUND] Directory {dir_name}: {image_count} images")

            if image_count > 0:
                data_sources.append({
                    'path': dir_name,
                    'type': 'directory',
                    'image_count': image_count,
                    'status': 'ready'
                })

    return data_sources

def extract_partial_zip(zip_path, extract_to='dataset', max_files=None):
    """
    Extract as much as possible from a partial zip file
    """
    print(f"[EXTRACT] Attempting to extract from {zip_path}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()

            # Filter for image files
            image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if max_files:
                image_files = image_files[:max_files]

            print(f"[EXTRACT] Found {len(image_files)} image files")

            if len(image_files) == 0:
                print("[WARNING] No image files found in archive")
                return 0

            # Create extraction directory
            os.makedirs(extract_to, exist_ok=True)

            # Extract files
            extracted_count = 0
            for file_path in image_files:
                try:
                    zip_ref.extract(file_path, extract_to)
                    extracted_count += 1

                    if extracted_count % 100 == 0:
                        print(f"[EXTRACT] Extracted {extracted_count}/{len(image_files)} images")

                except Exception as e:
                    print(f"[ERROR] Failed to extract {file_path}: {e}")
                    continue

            print(f"[SUCCESS] Extracted {extracted_count} images to {extract_to}")
            return extracted_count

    except zipfile.BadZipFile:
        print(f"[ERROR] {zip_path} is not a valid zip file")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to extract {zip_path}: {e}")
        return 0

def validate_images(directory, max_check=100):
    """
    Validate that images can be loaded properly
    """
    print(f"[VALIDATE] Checking images in {directory}...")

    image_paths = list(Path(directory).rglob('*.jpg')) + list(Path(directory).rglob('*.png'))

    if len(image_paths) == 0:
        print("[WARNING] No images found")
        return 0

    valid_count = 0
    check_count = min(max_check, len(image_paths))

    print(f"[VALIDATE] Checking {check_count} images...")

    for i, img_path in enumerate(image_paths[:check_count]):
        try:
            with Image.open(img_path) as img:
                img.verify()
            valid_count += 1

            if (i + 1) % 20 == 0:
                print(f"[VALIDATE] Checked {i + 1}/{check_count} images - {valid_count} valid")

        except Exception as e:
            print(f"[ERROR] Invalid image {img_path}: {e}")
            continue

    validity_rate = valid_count / check_count * 100
    print(f"[VALIDATE] {valid_count}/{check_count} images valid ({validity_rate:.1f}%)")

    return valid_count

def create_subset(source_dir, subset_dir, max_images=1000):
    """
    Create a smaller subset for initial training/testing
    """
    print(f"[SUBSET] Creating subset with {max_images} images...")

    os.makedirs(subset_dir, exist_ok=True)

    # Find all images
    image_paths = list(Path(source_dir).rglob('*.jpg')) + list(Path(source_dir).rglob('*.png'))

    if len(image_paths) < max_images:
        print(f"[WARNING] Only {len(image_paths)} images available, using all")
        max_images = len(image_paths)

    # Select subset
    selected_images = image_paths[:max_images]

    copied_count = 0
    for img_path in selected_images:
        try:
            # Create relative path structure
            rel_path = img_path.relative_to(source_dir)
            dest_path = Path(subset_dir) / rel_path

            # Create destination directory
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(img_path, dest_path)
            copied_count += 1

            if copied_count % 100 == 0:
                print(f"[SUBSET] Copied {copied_count}/{max_images} images")

        except Exception as e:
            print(f"[ERROR] Failed to copy {img_path}: {e}")
            continue

    print(f"[SUCCESS] Created subset with {copied_count} images in {subset_dir}")
    return copied_count

def prepare_training_data():
    """
    Main function to prepare data for training
    """
    print("[START] Dataset preparation for Stable Diffusion fine-tuning")
    print("=" * 60)

    # Check available data
    data_sources = check_partial_data()

    if len(data_sources) == 0:
        print("[ERROR] No datasets found!")
        print("[HELP] Download COCO dataset or place your images in a directory")
        return None

    print(f"\n[INFO] Found {len(data_sources)} data sources")

    # Try to work with available data
    training_data = None

    for source in data_sources:
        print(f"\n[PROCESS] Processing {source['path']} ({source['status']})")

        if source['type'] == 'directory' and source['status'] == 'ready':
            # Direct directory with images
            image_count = validate_images(source['path'])
            if image_count > 0:
                training_data = source['path']
                print(f"[SUCCESS] Using directory {source['path']} with {image_count} images")
                break

        elif source['type'] == 'zip' and source['status'] == 'complete':
            # Complete zip file
            extract_count = extract_partial_zip(source['path'], 'extracted_dataset')
            if extract_count > 0:
                validate_images('extracted_dataset')
                training_data = 'extracted_dataset'
                break

        elif source['type'] == 'partial_zip':
            # Partial zip - try to extract what we can
            print(f"[INFO] Attempting to extract partial data from {source['path']}")
            extract_count = extract_partial_zip(source['path'], 'partial_dataset', max_files=500)
            if extract_count > 0:
                validate_images('partial_dataset')
                training_data = 'partial_dataset'
                print("[INFO] Using partial dataset for initial testing")
                break

    # Create subset if we have too much data
    if training_data:
        image_count = len(list(Path(training_data).rglob('*.jpg'))) + len(list(Path(training_data).rglob('*.png')))

        if image_count > 2000:
            print(f"\n[INFO] Large dataset ({image_count} images), creating subset for initial training")
            subset_count = create_subset(training_data, 'training_subset', max_images=1000)
            training_data = 'training_subset'
            image_count = subset_count

    print("\n" + "=" * 60)
    if training_data:
        final_count = len(list(Path(training_data).rglob('*.jpg'))) + len(list(Path(training_data).rglob('*.png')))
        print(f"[SUCCESS] Dataset ready: {training_data}")
        print(f"[INFO] {final_count} images available for training")

        # Save dataset info
        dataset_info = {
            'path': training_data,
            'image_count': final_count,
            'type': 'subset' if 'subset' in training_data else 'full',
            'status': 'ready'
        }

        with open('dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)

        print("[SAVED] Dataset information saved to dataset_info.json")
        return dataset_info
    else:
        print("[ERROR] Could not prepare any dataset for training")
        return None

def main():
    dataset_info = prepare_training_data()

    if dataset_info:
        print("\n[NEXT] Ready to start training!")
        print("Run: python train_sd.py --instance_data_dir", dataset_info['path'])
    else:
        print("\n[WAIT] Please download more data or check your files")

if __name__ == "__main__":
    main()

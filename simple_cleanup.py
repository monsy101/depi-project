#!/usr/bin/env python3
"""
Simple project cleanup script
"""

import os
import shutil
from pathlib import Path

def cleanup():
    print("Starting project cleanup...")

    # Create main directories
    dirs = ["lora", "datasets", "results", "scripts", "config", "docker", "docs"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created {d}/")

    # Move LoRA if it exists
    if os.path.exists("lora_adapter"):
        shutil.move("lora_adapter", "lora/stable_diffusion_finetune_v1")
        print("Moved LoRA adapter")

    # Move scripts
    scripts = ["create_lora.py", "prepare_dataset.py", "quick_train.py",
               "train_sd.py", "test_model.py", "compare_models.py"]
    for script in scripts:
        if os.path.exists(script):
            shutil.move(script, f"scripts/{script}")
            print(f"Moved {script}")

    # Move images to results
    image_dirs = ["original_images", "finetuned_images"]
    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            # Create subdirectory in results
            sub_dir = img_dir.replace("_images", "")
            os.makedirs(f"results/{sub_dir}", exist_ok=True)

            # Move files
            for file in os.listdir(img_dir):
                shutil.move(os.path.join(img_dir, file), f"results/{sub_dir}")
            os.rmdir(img_dir)
            print(f"Moved {img_dir} to results/")

    # Move single images
    single_images = ["test_image.png", "lora_test_output.png", "model_comparison.png"]
    os.makedirs("results/test_images", exist_ok=True)
    os.makedirs("results/comparisons", exist_ok=True)

    for img in single_images:
        if os.path.exists(img):
            if "comparison" in img:
                dest = "results/comparisons"
            else:
                dest = "results/test_images"
            shutil.move(img, dest)
            print(f"Moved {img} to {dest}/")

    # Move datasets
    if os.path.exists("combined_dataset"):
        shutil.move("combined_dataset", "datasets/combined_dataset")
        print("Moved combined_dataset to datasets/")

    # Move config files
    config_files = ["requirements.txt", "minimal_training_config.json"]
    for conf in config_files:
        if os.path.exists(conf):
            shutil.move(conf, f"config/{conf}")
            print(f"Moved {conf} to config/")

    print("Cleanup completed!")

if __name__ == "__main__":
    cleanup()

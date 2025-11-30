#!/usr/bin/env python3
"""
Analyze partial zip file to extract available data
"""

import struct
import os

def read_zip_header(f):
    """Try to read zip file headers manually"""
    try:
        sig = f.read(4)
        if sig != b'PK\x03\x04':
            return None

        version = struct.unpack('<H', f.read(2))[0]
        flags = struct.unpack('<H', f.read(2))[0]
        compression = struct.unpack('<H', f.read(2))[0]
        mod_time = struct.unpack('<H', f.read(2))[0]
        mod_date = struct.unpack('<H', f.read(2))[0]
        crc32 = struct.unpack('<L', f.read(4))[0]
        compressed_size = struct.unpack('<L', f.read(4))[0]
        uncompressed_size = struct.unpack('<L', f.read(4))[0]
        filename_len = struct.unpack('<H', f.read(2))[0]
        extra_len = struct.unpack('<H', f.read(2))[0]

        filename = f.read(filename_len).decode('utf-8', errors='ignore')

        return {
            'filename': filename,
            'compressed_size': compressed_size,
            'uncompressed_size': uncompressed_size,
            'compression': compression
        }
    except:
        return None

def extract_file_from_zip(zip_path, file_info, output_dir):
    """Try to extract a single file from partial zip"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        with open(zip_path, 'rb') as f:
            # Skip to file data (this is approximate)
            # In a real implementation, we'd need more sophisticated parsing
            pass

        return False
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

def main():
    partial_file = 'train2014.6QhTQ3vM.zip.part'
    print(f'[ANALYSIS] Analyzing partial zip file: {partial_file}')

    if not os.path.exists(partial_file):
        print("[ERROR] Partial file not found")
        return

    file_size = os.path.getsize(partial_file)
    print(f"[INFO] File size: {file_size:,} bytes ({file_size/(1024**3):.2f} GB)")

    with open(partial_file, 'rb') as f:
        file_count = 0
        total_size = 0
        image_files = []

        print("[SCAN] Scanning for files...")

        while True:
            pos = f.tell()
            if pos >= file_size - 100:  # Near end of file
                break

            header = read_zip_header(f)

            if header is None:
                break

            file_count += 1
            total_size += header['uncompressed_size']

            filename = header['filename']
            print(f"File {file_count}: {filename} ({header['uncompressed_size']:,} bytes)")

            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(header)

            # Skip file data and extra field
            try:
                skip_size = header['compressed_size'] + header['extra_len']
                if skip_size > 0:
                    f.seek(skip_size, 1)
                else:
                    # If compressed_size is 0, might be a directory or special file
                    break
            except:
                break

            if file_count >= 50:  # Limit scan
                break

        print("\n[RESULTS]")
        print(f"Total files found: {file_count}")
        print(f"Image files: {len(image_files)}")
        print(f"Total uncompressed size: {total_size:,} bytes")

        if len(image_files) > 0:
            print("\n[EXTRACTION] Attempting to extract images...")
            extracted_count = 0

            for img_info in image_files[:20]:  # Extract first 20 images
                print(f"Extracting: {img_info['filename']}")
                # This would require more sophisticated zip parsing
                # For now, just count what we found

            print(f"\n[STATUS] Found {len(image_files)} images in partial archive")
            print("[NOTE] Full extraction would require complete zip file")
        else:
            print("[INFO] No image files found in scanned portion")

if __name__ == "__main__":
    main()

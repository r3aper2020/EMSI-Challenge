#!/usr/bin/env python3
import os
import sys
import json
import tarfile
import shutil
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Setup paths to import from backend app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.db.database import Database

BUCKET_NAME = 'rareplanes-public'

# S3 keys for the tiled dataset tarballs
TARBALL_KEYS = {
    "train_images": "real/tarballs/train/RarePlanes_train_PS-RGB_tiled.tar.gz",
    "train_geojson": "real/tarballs/train/RarePlanes_train_geojson_aircraft_tiled.tar.gz",
    "test_images": "real/tarballs/test/RarePlanes_test_PS-RGB_tiled.tar.gz",
    "test_geojson": "real/tarballs/test/RarePlanes_test_geojson_aircraft_tiled.tar.gz",
    "train_coco": "real/metadata_annotations/RarePlanes_Train_Coco_Annotations_tiled.json",
    "test_coco": "real/metadata_annotations/RarePlanes_Test_Coco_Annotations_tiled.json"
}

class ProgressCallback(object):
    def __init__(self, filename, size_bytes):
        self._filename = filename
        self._size = size_bytes
        self._seen_so_far = 0
        self._last_percent = -1
        
    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        if self._size > 0:
            percent = int((self._seen_so_far / self._size) * 100)
            if percent % 10 == 0 and percent != self._last_percent:
                print(f"  {self._filename}: {percent}% ({self._seen_so_far / 1024 / 1024:.1f} MB / {self._size / 1024 / 1024:.1f} MB)")
                self._last_percent = percent
        else:
            if self._seen_so_far % (10 * 1024 * 1024) == 0:
                print(f"  {self._filename}: {self._seen_so_far / 1024 / 1024:.1f} MB downloaded")

def download_file(s3_client, s3_key, local_path):
    if os.path.exists(local_path):
        print(f"File already downloaded: {local_path}")
        return
        
    print(f"Fetching s3://{BUCKET_NAME}/{s3_key}...")
    
    # Get file size
    response = s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
    size_bytes = response.get('ContentLength', 0)
    
    filename = os.path.basename(local_path)
    callback = ProgressCallback(filename, size_bytes)
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, s3_key, local_path, Callback=callback)
    print(f"Finished downloading {filename}.")

def extract_tarball(tar_path, extract_dir):
    print(f"Extracting {tar_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print(f"Extraction of {os.path.basename(tar_path)} complete.")

def convert_coco_to_yolo(coco_json_path, images_source_dir, output_images_dir, output_labels_dir, output_metadata_dir, dataset_type='real'):
    print(f"Parsing COCO JSON {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
        
    # Map image_id to annotations
    img_id_to_anns = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
        
    processed_count = 0
    skipped_count = 0
    
    # Process images
    for idx, img in enumerate(coco_data.get('images', [])):
        file_name = img['file_name']
        img_id = img['id']
        base_name = os.path.splitext(file_name)[0]
        
        # Check if the image file exists in the extracted directory
        src_img_path = os.path.join(images_source_dir, file_name)
        if not os.path.exists(src_img_path):
            # Try searching recursively in case it extracts to subdirs
            found = False
            for root, dirs, files in os.walk(images_source_dir):
                if file_name in files:
                    src_img_path = os.path.join(root, file_name)
                    found = True
                    break
            if not found:
                skipped_count += 1
                continue
                
        # Copy image file to target images folder
        dst_img_path = os.path.join(output_images_dir, file_name)
        if not os.path.exists(dst_img_path):
            shutil.copy2(src_img_path, dst_img_path)
            
        # Get annotations
        anns = img_id_to_anns.get(img_id, [])
        yolo_lines = []
        primary_ann = anns[0] if anns else {}
        
        for ann in anns:
            seg = ann.get('segmentation')
            if isinstance(seg, list):
                for poly in seg:
                    if len(poly) >= 6:
                        normalized_coords = []
                        for c_idx in range(0, len(poly), 2):
                            px = poly[c_idx] / img['width']
                            py = poly[c_idx+1] / img['height']
                            px = max(0.0, min(1.0, px))
                            py = max(0.0, min(1.0, py))
                            normalized_coords.append(f"{px:.5f} {py:.5f}")
                        # Class ID is 0 for aircraft
                        yolo_lines.append(f"0 {' '.join(normalized_coords)}")
                        
        # Save YOLO labels text file
        local_label_path = os.path.join(output_labels_dir, f"{base_name}.txt")
        with open(local_label_path, 'w') as lf:
            lf.write("\n".join(yolo_lines))
            
        # Create visualizer metadata
        location_str = primary_ann.get("location", "Unknown Airfield")
        airport_name = location_str.split(",")[0] if location_str else "Unknown Airfield"
        
        metadata = {
            "chip_id": base_name,
            "gsd_meters": 0.3, # WorldView-3 tiled is 0.3m
            "coordinate_system": "UTM Zone (WGS84)",
            "bounds": {
                "min_easting": 296000.0,
                "min_northing": 4312000.0,
                "max_easting": 296256.0,
                "max_northing": 4312256.0
            },
            "sensor_angle_deg": round(float(primary_ann.get("off_nadir_max", 15.0)), 2),
            "airport_code": airport_name,
            "time_of_capture": "2026-06-11T12:00:00Z",
            "scene_type": primary_ann.get("role", "apron")
        }
        
        # Save metadata JSON file
        local_meta_path = os.path.join(output_metadata_dir, f"{base_name}.json")
        with open(local_meta_path, 'w') as mf:
            json.dump(metadata, mf, indent=2)
            
        processed_count += 1
        if processed_count % 500 == 0:
            print(f"  Processed {processed_count} images so far...")
            
    print(f"Completed conversion: processed={processed_count}, missing/skipped={skipped_count}")
    return processed_count

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, "data", "cache")
    temp_extract_dir = os.path.join(base_dir, "data", "temp_extraction")
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    # Output structures
    output_dir = os.path.join(base_dir, "data", "rareplanes_real")
    output_images = os.path.join(output_dir, "images")
    output_labels = os.path.join(output_dir, "labels")
    output_metadata = os.path.join(output_dir, "metadata")
    
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    os.makedirs(output_metadata, exist_ok=True)
    
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # 1. Download tarballs and COCO JSON annotations
    print("=== Downloading RarePlanes Tiled Tarballs and COCO annotations from S3 ===")
    for name, key in TARBALL_KEYS.items():
        local_path = os.path.join(cache_dir, os.path.basename(key))
        download_file(s3, key, local_path)
        
    # 2. Extract tarballs
    print("\n=== Extracting Tarballs ===")
    for name in ["train_images", "train_geojson", "test_images", "test_geojson"]:
        tar_key = TARBALL_KEYS[name]
        tar_path = os.path.join(cache_dir, os.path.basename(tar_key))
        extract_dir = os.path.join(temp_extract_dir, name)
        extract_tarball(tar_path, extract_dir)
        
    # 3. Process train and test datasets, convert to YOLO, output under rareplanes_real
    print("\n=== Converting Annotations and Organizing Dataset Files ===")
    
    # Process Train Split
    train_coco_path = os.path.join(cache_dir, os.path.basename(TARBALL_KEYS["train_coco"]))
    train_images_src = os.path.join(temp_extract_dir, "train_images")
    train_count = convert_coco_to_yolo(
        train_coco_path, 
        train_images_src, 
        output_images, 
        output_labels, 
        output_metadata
    )
    
    # Process Test Split
    test_coco_path = os.path.join(cache_dir, os.path.basename(TARBALL_KEYS["test_coco"]))
    test_images_src = os.path.join(temp_extract_dir, "test_images")
    test_count = convert_coco_to_yolo(
        test_coco_path, 
        test_images_src, 
        output_images, 
        output_labels, 
        output_metadata
    )
    
    total_images = train_count + test_count
    print(f"\nDataset RarePlanes Real fully prepared with {total_images} total image chips.")
    
    # 4. Register dataset in database
    print("\n=== Registering Dataset in Workbench Database ===")
    db = Database()
    dataset_id = "dataset_rareplanes_real"
    desc = "Real Maxar WorldView-3 satellite imagery with aircraft annotations."
    
    if db.get_dataset(dataset_id):
        with db._get_conn() as conn:
            conn.execute(
                "UPDATE datasets SET sample_size = ?, folder_path = ? WHERE id = ?",
                (total_images, output_dir, dataset_id)
            )
            conn.commit()
        print(f"Updated registration for: {dataset_id}")
    else:
        db.create_dataset(
            dataset_id=dataset_id,
            name="RarePlanes Real",
            description=desc,
            task_type="instance_segmentation",
            sample_size=total_images,
            folder_path=output_dir
        )
        print(f"Created new registration for: {dataset_id}")
        
    # 5. Cleanup temporary extraction dir
    print("\n=== Cleaning Up Temporary Directories ===")
    try:
        shutil.rmtree(temp_extract_dir)
        print("Temporary extraction folder removed.")
    except Exception as ex:
        print(f"Warning: Could not clean up temporary directory {temp_extract_dir}: {ex}")
        
    print("\nDataset preparation completed successfully!")

if __name__ == '__main__':
    main()

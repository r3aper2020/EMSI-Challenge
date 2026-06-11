#!/usr/bin/env python3
import os
import sys
import json
import random
import argparse
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Setup paths to import from backend app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.db.database import Database

BUCKET_NAME = 'rareplanes-public'

def get_s3_client():
    return boto3.client('s3', config=Config(signature_version=UNSIGNED))

def download_file_with_cache(s3_client, s3_key, local_path):
    if os.path.exists(local_path):
        print(f"Using cached file: {local_path}")
        return
    
    print(f"Downloading s3://{BUCKET_NAME}/{s3_key} to {local_path}...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, s3_key, local_path)

def process_dataset(ds_type, count, base_dir, cache_dir):
    s3 = get_s3_client()
    
    output_dir = os.path.join(base_dir, "data", f"rareplanes_{ds_type}")
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    metadata_dir = os.path.join(output_dir, "metadata")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Define S3 paths and local cache paths
    if ds_type == 'real':
        ann_s3_key = 'real/metadata_annotations/RarePlanes_Train_Coco_Annotations_tiled.json'
        ann_cache_path = os.path.join(cache_dir, 'RarePlanes_Train_Coco_Annotations_tiled.json')
        s3_images_prefix = 'real/train/PS-RGB_tiled/'
        gsd = 0.3
    else:
        ann_s3_key = 'synthetic/metadata_annotations/instances_train_aircraft.json'
        ann_cache_path = os.path.join(cache_dir, 'instances_train_aircraft.json')
        s3_images_prefix = 'synthetic/train/images/'
        gsd = 0.5
        
    # Download annotations JSON
    download_file_with_cache(s3, ann_s3_key, ann_cache_path)
    
    print(f"Loading {ds_type} COCO annotations from JSON...")
    with open(ann_cache_path, 'r') as f:
        coco_data = json.load(f)
        
    # Map image_id to annotations
    print("Mapping annotations to image files...")
    img_id_to_anns = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
        
    # Filter for images that actually contain annotations (plane targets)
    images_with_anns = [
        img for img in coco_data.get('images', [])
        if img['id'] in img_id_to_anns and len(img_id_to_anns[img['id']]) > 0
    ]
    
    print(f"Found {len(images_with_anns)} images with airplane annotations.")
    
    # Shuffle and select subset
    random.seed(42)
    random.shuffle(images_with_anns)
    selected_images = images_with_anns[:count]
    
    print(f"Selected {len(selected_images)} images to download and process.")
    
    processed_count = 0
    for idx, img in enumerate(selected_images):
        file_name = img['file_name']
        img_id = str(img['id'])
        base_name = os.path.splitext(file_name)[0]
        
        # S3 image key and local path
        s3_img_key = f"{s3_images_prefix}{file_name}"
        local_img_path = os.path.join(images_dir, file_name)
        
        try:
            # Download image
            download_file_with_cache(s3, s3_img_key, local_img_path)
            
            # Process annotations to YOLO segmentations
            anns = img_id_to_anns[img['id']]
            yolo_lines = []
            
            # Extract first annotation details for metadata fallback
            primary_ann = anns[0] if anns else {}
            
            for ann in anns:
                seg = ann.get('segmentation')
                if isinstance(seg, list):
                    for poly in seg:
                        if len(poly) >= 6: # Need at least 3 points
                            normalized_coords = []
                            for c_idx in range(0, len(poly), 2):
                                px = poly[c_idx] / img['width']
                                py = poly[c_idx+1] / img['height']
                                px = max(0.0, min(1.0, px))
                                py = max(0.0, min(1.0, py))
                                normalized_coords.append(f"{px:.5f} {py:.5f}")
                            yolo_lines.append(f"0 {' '.join(normalized_coords)}")
                            
            # Save YOLO text file
            local_label_path = os.path.join(labels_dir, f"{base_name}.txt")
            with open(local_label_path, 'w') as lf:
                lf.write("\n".join(yolo_lines))
                
            # Create metadata JSON for visualization
            airport_name = "Unknown Airport"
            if ds_type == 'real':
                location_str = primary_ann.get("location", "Unknown Airport")
                airport_name = location_str.split(",")[0] if location_str else "Unknown Airport"
                role_val = primary_ann.get("role", "apron")
            else:
                # Synthetic name parsing
                parts = file_name.split("_")
                if len(parts) > 0:
                    airport_name = parts[0] + " Airport"
                role_val = primary_ann.get("full", "Airplane_Civil_Transport").split("_")[-1]
                
            metadata = {
                "chip_id": base_name,
                "gsd_meters": gsd,
                "coordinate_system": "UTM Zone 18N (WGS84)" if ds_type == 'real' else "WGS84",
                "bounds": {
                    "min_easting": 296000.0 + random.uniform(-500, 500),
                    "min_northing": 4312000.0 + random.uniform(-500, 500),
                    "max_easting": 296256.0,
                    "max_northing": 4312256.0
                },
                "sensor_angle_deg": round(float(primary_ann.get("off_nadir_max", random.uniform(5, 20))), 2),
                "airport_code": airport_name,
                "time_of_capture": f"2026-06-11T{random.randint(8,17):02d}:{random.randint(0,59):02d}:00Z",
                "scene_type": role_val
            }
            
            # Save metadata JSON file
            local_meta_path = os.path.join(metadata_dir, f"{base_name}.json")
            with open(local_meta_path, 'w') as mf:
                json.dump(metadata, mf, indent=2)
                
            processed_count += 1
            print(f"[{idx+1}/{len(selected_images)}] Processed {file_name} successfully.")
        except Exception as ex:
            print(f"Error processing {file_name}: {ex}")
            
    print(f"Successfully prepared {processed_count} images for dataset: rareplanes_{ds_type}")
    return processed_count

def main():
    parser = argparse.ArgumentParser(description="Download and prep RarePlanes public S3 dataset")
    parser.add_argument('--real-count', type=int, default=50, help="Number of real images to download")
    parser.add_argument('--synthetic-count', type=int, default=50, help="Number of synthetic images to download")
    parser.add_argument('--skip-real', action='store_true', help="Skip downloading real dataset")
    parser.add_argument('--skip-synthetic', action='store_true', help="Skip downloading synthetic dataset")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    db = Database()
    
    if not args.skip_real and args.real_count > 0:
        print("\n=== Processing RarePlanes Real ===")
        real_processed = process_dataset('real', args.real_count, base_dir, cache_dir)
        
        # Register in Database
        dataset_id = "dataset_rareplanes_real"
        desc = "Real Maxar WorldView-3 satellite imagery with aircraft segmentations."
        folder_path = os.path.join(base_dir, "data", "rareplanes_real")
        
        if db.get_dataset(dataset_id):
            with db._get_conn() as conn:
                conn.execute(
                    "UPDATE datasets SET sample_size = ?, folder_path = ? WHERE id = ?",
                    (real_processed, folder_path, dataset_id)
                )
                conn.commit()
            print(f"Updated registration for dataset: {dataset_id}")
        else:
            db.create_dataset(
                dataset_id=dataset_id,
                name="RarePlanes Real",
                description=desc,
                task_type="instance_segmentation",
                sample_size=real_processed,
                folder_path=folder_path
            )
            print(f"Created new registration for dataset: {dataset_id}")
            
    if not args.skip_synthetic and args.synthetic_count > 0:
        print("\n=== Processing RarePlanes Synthetic ===")
        syn_processed = process_dataset('synthetic', args.synthetic_count, base_dir, cache_dir)
        
        # Register in Database
        dataset_id = "dataset_rareplanes_synthetic"
        desc = "Procedural geospatial synthetic airport dataset generated via AI.Reverie platform."
        folder_path = os.path.join(base_dir, "data", "rareplanes_synthetic")
        
        if db.get_dataset(dataset_id):
            with db._get_conn() as conn:
                conn.execute(
                    "UPDATE datasets SET sample_size = ?, folder_path = ? WHERE id = ?",
                    (syn_processed, folder_path, dataset_id)
                )
                conn.commit()
            print(f"Updated registration for dataset: {dataset_id}")
        else:
            db.create_dataset(
                dataset_id=dataset_id,
                name="RarePlanes Synthetic",
                description=desc,
                task_type="instance_segmentation",
                sample_size=syn_processed,
                folder_path=folder_path
            )
            print(f"Created new registration for dataset: {dataset_id}")

if __name__ == '__main__':
    main()

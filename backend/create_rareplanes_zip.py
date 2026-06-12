import os
import json
import tarfile
import zipfile
import shutil
import random

def main():
    print("Starting RarePlanes train dataset zip creator...")
    
    # Paths
    base_dir = "/Users/mwilliams/Desktop/dev/EMSI-Challenge"
    cache_dir = os.path.join(base_dir, "backend", "data", "cache")
    coco_path = os.path.join(cache_dir, "RarePlanes_Train_Coco_Annotations_tiled.json")
    tar_path = os.path.join(cache_dir, "RarePlanes_train_PS-RGB_tiled.tar.gz")
    
    output_dir = os.path.join(base_dir, "backend", "data", "rareplanes_real_subset")
    images_out_dir = os.path.join(output_dir, "images")
    labels_out_dir = os.path.join(output_dir, "labels")
    metadata_out_dir = os.path.join(output_dir, "metadata")
    
    # Clean up previous temporary directories
    shutil.rmtree(output_dir, ignore_errors=True)
    
    # Create directories
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)
    os.makedirs(metadata_out_dir, exist_ok=True)
    
    # 1. Load COCO JSON
    print("Loading COCO annotations...")
    with open(coco_path, "r") as f:
        coco_data = json.load(f)
        
    print(f"Loaded {len(coco_data.get('images', []))} images and {len(coco_data.get('annotations', []))} annotations.")
    
    # Map image ID to image object
    img_map = {img["id"]: img for img in coco_data.get("images", [])}
    
    # Group annotations by image ID
    annotations_by_img = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        annotations_by_img.setdefault(img_id, []).append(ann)
        
    # Get set of all annotated image filenames (mapped by basename)
    annotated_basenames = {}
    for img_id, anns in annotations_by_img.items():
        if len(anns) > 0 and img_id in img_map:
            img_info = img_map[img_id]
            file_name = img_info["file_name"]
            basename = os.path.basename(file_name)
            annotated_basenames[basename] = (img_id, img_info, anns)
            
    print(f"Total annotated images available: {len(annotated_basenames)}")
    
    # 2. Iterate through tarball and extract 100 annotated images
    print("Extracting subset from tarball (optimized search)...")
    tar = tarfile.open(tar_path, "r:gz")
    
    extracted_count = 0
    target_count = 400
    
    # UTM coordinates base
    base_easting = 296278.85
    base_northing = 4311050.02
    width_meters = 256.0
    
    for member in tar:
        if member.isfile() and member.name.lower().endswith((".png", ".jpg", ".jpeg")):
            tar_basename = os.path.basename(member.name)
            
            # Check if this file is in our annotated images
            match_key = None
            if tar_basename in annotated_basenames:
                match_key = tar_basename
            else:
                # Try fuzzy matching in case of minor path mismatch
                for key in annotated_basenames:
                    if key in tar_basename or tar_basename in key:
                        match_key = key
                        break
            
            if match_key:
                img_id, img_info, anns = annotated_basenames[match_key]
                
                # We found a match! Extract it.
                img_data = tar.extractfile(member)
                if img_data is None:
                    continue
                    
                # Format name matching rareplanes_test.zip structure: rareplanes_chip_XXX
                chip_name = f"rareplanes_chip_{extracted_count:03d}"
                
                # Save image file as PNG
                out_img_path = os.path.join(images_out_dir, f"{chip_name}.png")
                with open(out_img_path, "wb") as out_f:
                    out_f.write(img_data.read())
                    
                # Process annotations to YOLO polygon segmentations
                yolo_lines = []
                width = img_info.get("width", 512)
                height = img_info.get("height", 512)
                
                for ann in anns:
                    role = ann.get("role", "")
                    if role in ['Small Civil Transport/Utility', 'Military Fighter/Interceptor/Attack', 'Military Trainer']:
                        class_idx = 0  # Small Aircraft
                    elif role in ['Medium Civil Transport/Utility', 'Military Transport/Utility/AWAC']:
                        class_idx = 1  # Cargo Plane
                    elif role in ['Large Civil Transport/Utility', 'Military Bomber']:
                        class_idx = 2  # Large Aircraft
                    else:
                        class_idx = 0  # Default fallback to Small Aircraft
                        
                    seg = ann.get("segmentation")
                    polygon = []
                    
                    if seg and isinstance(seg, list) and len(seg) > 0:
                        if isinstance(seg[0], list):
                            polygon = seg[0]
                        else:
                            polygon = seg
                            
                    if len(polygon) >= 6:
                        normalized_poly = []
                        for i in range(0, len(polygon), 2):
                            px = polygon[i] / width
                            py = polygon[i+1] / height
                            px = max(0.0, min(1.0, px))
                            py = max(0.0, min(1.0, py))
                            normalized_poly.append(f"{px:.6f} {py:.6f}")
                        poly_str = " ".join(normalized_poly)
                        yolo_lines.append(f"{class_idx} {poly_str}")
                    else:
                        bbox = ann.get("bbox")
                        if bbox and len(bbox) == 4:
                            x_min, y_min, w, h = bbox
                            corners = [
                                x_min, y_min,
                                x_min + w, y_min,
                                x_min + w, y_min + h,
                                x_min, y_min + h
                            ]
                            normalized_poly = []
                            for i in range(0, len(corners), 2):
                                px = corners[i] / width
                                py = corners[i+1] / height
                                px = max(0.0, min(1.0, px))
                                py = max(0.0, min(1.0, py))
                                normalized_poly.append(f"{px:.6f} {py:.6f}")
                            poly_str = " ".join(normalized_poly)
                            yolo_lines.append(f"{class_idx} {poly_str}")
                            
                # Save label file
                out_label_path = os.path.join(labels_out_dir, f"{chip_name}.txt")
                with open(out_label_path, "w") as lbl_f:
                    lbl_f.write("\n".join(yolo_lines) + "\n")
                    
                # Create metadata matching rareplanes_test.zip format exactly
                min_e = base_easting + extracted_count * 300.0
                min_n = base_northing + extracted_count * 300.0
                
                meta_dict = {
                    "chip_id": chip_name,
                    "gsd_meters": float(img_info.get("gsd") or 0.5),
                    "coordinate_system": "UTM Zone 18N (WGS84)",
                    "bounds": {
                        "min_easting": min_e,
                        "min_northing": min_n,
                        "max_easting": min_e + width_meters,
                        "max_northing": min_n + width_meters
                    },
                    "sensor_angle_deg": round(float(img_info.get("incidence_angle") or 4.13 + random.uniform(-1, 1)), 2),
                    "airport_code": "KIAD",
                    "time_of_capture": f"2026-06-11T11:{random.randint(10, 59):02d}:00Z",
                    "scene_type": random.choice(["apron", "taxiway", "runway"]),
                    "sensor_type": "Optical (WorldView-3)",
                    "sar_polarization": "N/A",
                    "incidence_angle": 0.0
                }
                
                out_meta_path = os.path.join(metadata_out_dir, f"{chip_name}.json")
                with open(out_meta_path, "w") as meta_f:
                    json.dump(meta_dict, meta_f, indent=2)
                    
                extracted_count += 1
                if extracted_count % 10 == 0:
                    print(f"Extracted {extracted_count}/{target_count} images...")
                    
                if extracted_count >= target_count:
                    break
                    
    tar.close()
    print(f"Finished extracting {extracted_count} images.")
    
    if extracted_count < target_count:
        print(f"Warning: Only extracted {extracted_count} images.")
        
    # 3. Create ZIP files
    zip_filenames = ["rareplanes_train.zip"]
    for zip_name in zip_filenames:
        zip_path = os.path.join(base_dir, "backend", zip_name)
        print(f"Creating ZIP archive at {zip_path}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
        print(f"ZIP file created successfully: {zip_path}")
        
        # Copy to the root of the workspace for easy user access
        workspace_copy_path = os.path.join(base_dir, zip_name)
        shutil.copy2(zip_path, workspace_copy_path)
        print(f"Workspace copy created at: {workspace_copy_path}")
        
    # Cleanup output directory
    shutil.rmtree(output_dir, ignore_errors=True)
    print("Cleaned up temporary folders.")
    print("Done!")

if __name__ == "__main__":
    main()

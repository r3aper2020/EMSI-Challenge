import os
import shutil
import random
import yaml
import json
import time

STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "storage")

class DatasetService:
    def __init__(self, db):
        self.db = db
        self.storage_dir = STORAGE_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(os.path.join(self.storage_dir, "dataset_exports"), exist_ok=True)

    def scan_and_register_all_datasets(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_dir = os.path.join(base_dir, "data")
        if not os.path.exists(data_dir):
            return []
            
        registered = []
        for item in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, item)
            if not os.path.isdir(folder_path) or item in ("storage", "cache", "venv"):
                continue
                
            images_dir = os.path.join(folder_path, "images")
            if not os.path.exists(images_dir):
                continue
                
            img_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            sample_size = len(img_files)
            
            dataset_id = f"dataset_{item}"
            dataset = self.db.get_dataset(dataset_id)
            
            name = item.replace("_", " ").title()
            if "sample" in item.lower():
                desc = "Procedural geospatial synthetic airport dataset for aircraft instance segmentation (Sample)"
            elif "real" in item.lower():
                desc = "Real Maxar WorldView-3 satellite imagery with aircraft annotations"
            elif "synthetic" in item.lower():
                desc = "Procedural geospatial synthetic airport dataset generated via AI.Reverie platform"
            else:
                desc = f"Geospatial airfield dataset: {name}"
                
            if not dataset:
                dataset = self.db.create_dataset(
                    dataset_id=dataset_id,
                    name=name,
                    description=desc,
                    task_type="instance_segmentation",
                    sample_size=sample_size,
                    folder_path=folder_path
                )
            else:
                with self.db._get_conn() as conn:
                    conn.execute("UPDATE datasets SET sample_size = ?, folder_path = ? WHERE id = ?", (sample_size, folder_path, dataset_id))
                    conn.commit()
                dataset = self.db.get_dataset(dataset_id)
                
            registered.append(dataset)
        return registered


    def create_split_and_export(self, dataset_id, version_tag, train_ratio=0.8, val_ratio=0.2, seed=42):
        dataset = self.db.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        # Register new version in DB with status "exporting"
        version_id = f"version_{dataset_id}_{int(time.time())}"
        self.db.create_dataset_version(
            version_id=version_id,
            dataset_id=dataset_id,
            version_tag=version_tag,
            train_split=train_ratio,
            val_split=val_ratio,
            split_seed=seed,
            status="exporting"
        )
        
        try:
            source_dir = dataset["folder_path"]
            images_dir = os.path.join(source_dir, "images")
            labels_dir = os.path.join(source_dir, "labels")
            metadata_dir = os.path.join(source_dir, "metadata")
            
            img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            
            # Shuffle list based on seed
            random.seed(seed)
            random.shuffle(img_files)
            
            total_count = len(img_files)
            train_count = int(total_count * train_ratio)
            
            train_imgs = img_files[:train_count]
            val_imgs = img_files[train_count:]
            
            # Create export directory structure
            export_path = os.path.join(self.storage_dir, "dataset_exports", version_id)
            for split in ["train", "val"]:
                os.makedirs(os.path.join(export_path, split, "images"), exist_ok=True)
                os.makedirs(os.path.join(export_path, split, "labels"), exist_ok=True)
            
            manifest = {
                "train": [],
                "val": []
            }
            
            # Helper to copy images, labels and gather metadata
            def copy_split_data(images_list, split_name):
                for img_name in images_list:
                    base_name = os.path.splitext(img_name)[0]
                    # Source paths
                    src_img = os.path.join(images_dir, img_name)
                    src_lbl = os.path.join(labels_dir, f"{base_name}.txt")
                    src_meta = os.path.join(metadata_dir, f"{base_name}.json")
                    
                    # Target paths
                    dst_img = os.path.join(export_path, split_name, "images", img_name)
                    dst_lbl = os.path.join(export_path, split_name, "labels", f"{base_name}.txt")
                    
                    # Try to use hard link for instant export and zero disk duplication
                    try:
                        if os.path.exists(dst_img):
                            os.remove(dst_img)
                        os.link(src_img, dst_img)
                    except Exception:
                        shutil.copy2(src_img, dst_img)

                    if os.path.exists(src_lbl):
                        try:
                            if os.path.exists(dst_lbl):
                                os.remove(dst_lbl)
                            os.link(src_lbl, dst_lbl)
                        except Exception:
                            shutil.copy2(src_lbl, dst_lbl)
                    else:
                        # Write empty label file if none exists (negative background sample)
                        open(dst_lbl, 'w').close()
                        
                    # Load metadata
                    meta_content = {}
                    if os.path.exists(src_meta):
                        with open(src_meta, "r") as mf:
                            meta_content = json.load(mf)
                            
                    manifest[split_name].append({
                        "image_id": base_name,
                        "image_path": dst_img,
                        "label_path": dst_lbl,
                        "metadata": meta_content
                    })
            
            copy_split_data(train_imgs, "train")
            copy_split_data(val_imgs, "val")
            
            # Write dataset.yaml for YOLO
            dataset_yaml = {
                "path": export_path,
                "train": "train/images",
                "val": "val/images",
                "names": {
                    0: "aircraft"
                }
            }
            
            yaml_path = os.path.join(export_path, "dataset.yaml")
            with open(yaml_path, "w") as yf:
                yaml.dump(dataset_yaml, yf, default_flow_style=False)
                
            # Update DB with complete status and export path
            self.db.create_dataset_version(
                version_id=version_id,
                dataset_id=dataset_id,
                version_tag=version_tag,
                train_split=train_ratio,
                val_split=val_ratio,
                split_seed=seed,
                status="complete",
                export_path=export_path,
                manifest=manifest
            )
            return self.db.get_dataset_version(version_id)
            
        except Exception as e:
            # Mark version as failed
            self.db.create_dataset_version(
                version_id=version_id,
                dataset_id=dataset_id,
                version_tag=version_tag,
                train_split=train_ratio,
                val_split=val_ratio,
                split_seed=seed,
                status=f"failed: {str(e)}"
            )
            raise e
            
    def get_version_details(self, version_id):
        return self.db.get_dataset_version(version_id)

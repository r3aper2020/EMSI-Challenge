import os
import sys
import uuid
import time
import zipfile
import io
import shutil
import random

# Load .env file manually if it exists
for env_path in [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),  # backend/.env
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),  # root/.env
]:
    if os.path.exists(env_path):
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip().strip("'").strip('"')
            print(f"Loaded environment variables from: {env_path}")
        except Exception as e:
            print(f"Failed to load env file {env_path}: {e}")

from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.db.database import Database
from app.services.dataset_service import DatasetService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.workers.training_worker import TrainingWorker

# Initialize FastAPI
app = FastAPI(title="ATR Model Production Workbench API", version="1.0.0")

# CORS middleware to allow connection from Vite development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database and Services
db = Database()
dataset_service = DatasetService(db)
embedding_service = EmbeddingService(db)
llm_service = LLMService(db)

# Start background training worker
worker = None

@app.on_event("startup")
def startup_event():
    global worker
    worker = TrainingWorker()
    worker.start()
    
    # Auto-initialize all datasets on startup
    try:
        datasets = dataset_service.scan_and_register_all_datasets()
        for ds in datasets:
            print(f"Dataset registered: {ds['id']}")
            # Proactively generate embeddings for visualization if they don't exist
            embs = db.get_embeddings_by_dataset(ds["id"])
            if not embs:
                print(f"Generating embeddings for dataset: {ds['id']}...")
                embedding_service.generate_and_save_embeddings(ds["id"])
    except Exception as e:
        print(f"Failed to auto-initialize datasets on startup: {e}")

@app.on_event("shutdown")
def shutdown_event():
    global worker
    if worker:
        worker.stop()

# Dynamic images endpoint to serve images from any registered dataset directory
@app.get("/static/images/{image_name}")
def get_image_file(image_name: str):
    image_id = os.path.splitext(image_name)[0]
    
    # 1. Try to resolve image path from the database registered datasets
    try:
        with db._get_conn() as conn:
            row = conn.execute("SELECT dataset_id FROM embeddings WHERE image_id = ?", (image_id,)).fetchone()
            if row:
                dataset_id = row[0]
                ds = db.get_dataset(dataset_id)
                if ds:
                    img_path = os.path.join(ds["folder_path"], "images", image_name)
                    if os.path.exists(img_path):
                        return FileResponse(img_path)
    except Exception as e:
        print(f"Error resolving image path from DB: {e}")
        
    # 2. Fallback scan of directories in backend/data/
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    if os.path.exists(base_data_dir):
        for ds_dir in os.listdir(base_data_dir):
            ds_path = os.path.join(base_data_dir, ds_dir)
            if os.path.isdir(ds_path) and ds_dir not in ("storage", "cache", "venv"):
                img_path = os.path.join(ds_path, "images", image_name)
                if os.path.exists(img_path):
                    return FileResponse(img_path)
                    
    raise HTTPException(status_code=404, detail=f"Image {image_name} not found")

# Pydantic schemas
class ProjectCreate(BaseModel):
    name: str
    description: str = ""

class DatasetSplitRequest(BaseModel):
    dataset_id: str
    version_tag: str
    train_split: float = 0.8
    val_split: float = 0.2
    split_seed: int = 42

class ExperimentCreate(BaseModel):
    project_id: str
    dataset_version_id: str
    name: str
    task_type: str = "instance_segmentation"
    model_type: str = "yolo_seg"
    config: dict = {}
    pipeline_id: str = None

class LLMCommandRequest(BaseModel):
    prompt: str

class LLMCommandSaveRequest(BaseModel):
    sender: str
    text: str
    payload: dict = None

class SaveLabelsRequest(BaseModel):
    labels: list

# --- API Endpoints ---

@app.get("/api/health")
def health_check():
    return {"status": "ok", "worker_alive": worker.is_alive() if worker else False}

@app.post("/api/initialize")
def initialize_workbench():
    """Manually triggers scanning, dataset registration, and PCA embeddings extraction."""
    try:
        # Scan and register all available datasets (which will locate rareplanes_real)
        datasets = dataset_service.scan_and_register_all_datasets()
        real_ds = None
        for ds in datasets:
            if ds["id"] == "dataset_rareplanes_real":
                real_ds = ds
                break
                
        if not real_ds:
            raise HTTPException(
                status_code=404, 
                detail="Real RarePlanes dataset directory not found at backend/data/rareplanes_real."
            )
            
        embs = embedding_service.generate_and_save_embeddings(real_ds["id"])
        
        # Verify if a default project exists, create if not
        project_id = "proj_default"
        proj = db.get_project(project_id)
        if not proj:
            db.create_project(project_id, "Geospatial Target Recognition", "Workbench project for managing and training ATR models on airfield datasets.")
            
        return {
            "status": "success",
            "dataset": real_ds,
            "embeddings_count": len(embs),
            "project_id": project_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects")
def list_projects():
    return db.get_projects()

@app.post("/api/projects")
def create_project(data: ProjectCreate):
    proj_id = f"proj_{int(time.time())}"
    return db.create_project(proj_id, data.name, data.description)

@app.get("/api/projects/{project_id}/tree")
def get_project_tree_endpoint(project_id: str):
    tree = db.get_project_tree(project_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Project not found")
    return tree

@app.get("/api/projects/{project_id}/pipelines")
def list_project_pipelines(project_id: str):
    return db.get_pipelines(project_id)

@app.post("/api/pipelines/{pipeline_id}/experiments")
def trigger_pipeline_experiment(pipeline_id: str, data: dict = Body(...)):
    pipeline = db.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
        
    exp_id = f"exp_{int(time.time())}"
    
    # Determine name
    name = data.get("name")
    if not name:
        cursor = db.get_experiments()
        runs_in_pipe = [e for e in cursor if e.get("pipeline_id") == pipeline_id]
        run_number = len(runs_in_pipe) + 1
        
        # Default description for non-training runs
        if pipeline["id"] == "pipe_curation":
            name = f"Run {run_number:03d}: Outliers & duplicates curation"
        elif pipeline["id"] == "pipe_labeling_review":
            name = f"Run {run_number:03d}: Active learning label proposals"
        elif pipeline["id"] == "pipe_eval_review":
            name = f"Run {run_number:03d}: Ground truth vs predictions review"
        else:
            name = f"Run {run_number:03d}"
            
    # Create curation/review experiment directly
    exp = db.create_experiment(
        exp_id=exp_id,
        project_id="proj_default",
        dataset_version_id=data.get("dataset_version_id"),
        name=name,
        task_type=pipeline["type"],
        model_type=pipeline["type"],
        config_dict=data.get("config", {}),
        pipeline_id=pipeline_id
    )
    
    # Simulate execution status: start as running, and mark complete after 3 seconds
    db.update_experiment_status(exp_id, "running")
    
    import threading
    def simulate_non_training_workflow():
        time.sleep(3)
        db.update_experiment_status(exp_id, "complete")
        
    threading.Thread(target=simulate_non_training_workflow, daemon=True).start()
    
    return exp



@app.get("/api/datasets")
def list_datasets():
    return db.get_datasets()

@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    ds = db.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds

@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    success = dataset_service.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"status": "success", "message": "Dataset successfully deleted."}

def deterministic_hash(s: str) -> int:
    h = 0
    for char in s:
        h = (h * 31 + ord(char)) & 0xFFFFFFFF
    return h

@app.get("/api/datasets/{dataset_id}/embeddings")
def get_dataset_embeddings(dataset_id: str):
    embeddings = db.get_embeddings_by_dataset(dataset_id)
    
    # Locate dataset path to read label files
    ds = db.get_dataset(dataset_id)
    if ds and ds.get("folder_path"):
        labels_dir = os.path.join(ds["folder_path"], "labels")
    else:
        labels_dir = None
            
    # For each embedding, read labels if label file exists
    for emb in embeddings:
        image_id = emb["image_id"]
        
        # Inject deterministic SAR vs Optical parameters for dataset exploration
        h_val = deterministic_hash(image_id)
        if "metadata" not in emb or not emb["metadata"]:
            emb["metadata"] = {}
            
        if h_val % 2 == 0:
            emb["metadata"]["sensor_type"] = "SAR"
            pols = ["VV", "VH", "VV+VH"]
            emb["metadata"]["sar_polarization"] = pols[h_val % len(pols)]
            emb["metadata"]["incidence_angle"] = round(18.5 + (h_val % 235) / 10.0, 1) # 18.5 to 42.0
        else:
            emb["metadata"]["sensor_type"] = "Optical (WorldView-3)"
            emb["metadata"]["sar_polarization"] = "N/A"
            emb["metadata"]["incidence_angle"] = 0.0

        parsed_labels = []
        if labels_dir and os.path.exists(labels_dir):
            label_file = os.path.join(labels_dir, f"{image_id}.txt")
            if os.path.exists(label_file):
                try:
                    with open(label_file, "r") as lf:
                        lines = lf.readlines()
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            tokens = line.split()
                            if len(tokens) >= 5: # class_id + at least 2 coordinate pairs
                                class_id = int(tokens[0])
                                coords = [float(t) for t in tokens[1:]]
                                # Reshape into pairs of coordinates
                                polygon = []
                                for idx in range(0, len(coords) - 1, 2):
                                    polygon.append([coords[idx], coords[idx+1]])
                                
                                if len(polygon) >= 2:
                                    xs = [pt[0] for pt in polygon]
                                    ys = [pt[1] for pt in polygon]
                                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                                    parsed_labels.append({
                                        "class_id": class_id,
                                        "polygon": polygon,
                                        "bbox": bbox
                                    })
                except Exception as e:
                    print(f"Error parsing label file {label_file}: {e}")
        emb["labels"] = parsed_labels
            
    return embeddings

@app.post("/api/datasets/{dataset_id}/images/{image_id}/auto-label")
def auto_label_image(dataset_id: str, image_id: str):
    """Simulates active model inference suggestion queue for geospatial targets (SME-in-the-loop)."""
    ds = db.get_dataset(dataset_id)
    if not ds or not ds.get("folder_path"):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    labels_dir = os.path.join(ds["folder_path"], "labels")
    label_file = os.path.join(labels_dir, f"{image_id}.txt")
    
    suggestions = []
    
    # 1. Check if there are existing labels. If so, perturb them slightly to mock auto-detections.
    existing_labels = []
    if os.path.exists(label_file):
        try:
            with open(label_file, "r") as lf:
                for line in lf.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    tokens = line.split()
                    if len(tokens) >= 5:
                        class_id = int(tokens[0])
                        coords = [float(t) for t in tokens[1:]]
                        polygon = []
                        for idx in range(0, len(coords) - 1, 2):
                            polygon.append([coords[idx], coords[idx+1]])
                        if len(polygon) >= 2:
                            existing_labels.append({"class_id": class_id, "polygon": polygon})
        except Exception as e:
            print(f"Error loading existing labels for auto-labeling: {e}")
            
    if existing_labels:
        # Perturb existing labels slightly
        rnd = random.Random(deterministic_hash(image_id))
        for idx, lbl in enumerate(existing_labels):
            scale = 0.012
            new_poly = []
            for pt in lbl["polygon"]:
                dx = rnd.uniform(-scale, scale)
                dy = rnd.uniform(-scale, scale)
                new_poly.append([
                    max(0.0, min(1.0, pt[0] + dx)),
                    max(0.0, min(1.0, pt[1] + dy))
                ])
            
            xs = [pt[0] for pt in new_poly]
            ys = [pt[1] for pt in new_poly]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            conf = round(rnd.uniform(0.78, 0.96), 2)
            
            suggestions.append({
                "id": f"sugg_{idx}_{int(time.time())}",
                "class_id": lbl["class_id"],
                "polygon": new_poly,
                "bbox": bbox,
                "confidence": conf
            })
    else:
        # 2. No existing labels: return 3 mock target suggestions placed realistically in the chip
        rnd = random.Random(deterministic_hash(image_id))
        mock_polys = [
            [[0.22, 0.45], [0.28, 0.41], [0.34, 0.45], [0.28, 0.49]],
            [[0.65, 0.22], [0.71, 0.18], [0.77, 0.22], [0.71, 0.26]],
            [[0.45, 0.75], [0.51, 0.71], [0.57, 0.75], [0.51, 0.79]]
        ]
        
        for idx, poly in enumerate(mock_polys):
            conf = round(rnd.uniform(0.74, 0.94), 2)
            # Add a slight random noise to template coordinates
            perturbed_poly = []
            for pt in poly:
                dx = rnd.uniform(-0.01, 0.01)
                dy = rnd.uniform(-0.01, 0.01)
                perturbed_poly.append([max(0.0, min(1.0, pt[0] + dx)), max(0.0, min(1.0, pt[1] + dy))])
                
            xs = [pt[0] for pt in perturbed_poly]
            ys = [pt[1] for pt in perturbed_poly]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            
            suggestions.append({
                "id": f"sugg_{idx}_{int(time.time())}",
                "class_id": 0, # Aircraft class
                "polygon": perturbed_poly,
                "bbox": bbox,
                "confidence": conf
            })
            
    return {"status": "success", "suggestions": suggestions}

@app.post("/api/datasets/{dataset_id}/images/{image_id}/labels")
def save_dataset_image_labels(dataset_id: str, image_id: str, req: SaveLabelsRequest):
    ds = db.get_dataset(dataset_id)
    if not ds or not ds.get("folder_path"):
        raise HTTPException(status_code=404, detail="Dataset not found")
    labels_dir = os.path.join(ds["folder_path"], "labels")
        
    os.makedirs(labels_dir, exist_ok=True)
    label_file = os.path.join(labels_dir, f"{image_id}.txt")
    
    try:
        with open(label_file, "w") as f:
            lines = []
            for label in req.labels:
                class_id = label.get("class_id", 0)
                polygon = label.get("polygon", [])
                if not polygon:
                    continue
                # Flatten the polygon coords
                coords_str = " ".join([f"{pt[0]:.5f} {pt[1]:.5f}" for pt in polygon])
                lines.append(f"{class_id} {coords_str}")
            f.write("\n".join(lines))
        return {"status": "success", "message": "Labels saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/datasets/{dataset_id}/images/{image_id}")
def get_dataset_image(dataset_id: str, image_id: str):
    # Determine the directory path
    ds = db.get_dataset(dataset_id)
    if ds and ds.get("folder_path"):
        images_dir = os.path.join(ds["folder_path"], "images")
    else:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        images_dir = os.path.join(base_dir, "data", "storage", "raw_datasets", dataset_id, "images")
        
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail="Dataset folder not found")
        
    # Search for image matching image_id with valid extensions
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        filename = f"{image_id}{ext}"
        path = os.path.join(images_dir, filename)
        if os.path.exists(path):
            return FileResponse(path)
            
    raise HTTPException(status_code=404, detail=f"Image {image_id} not found in {dataset_id}")

@app.post("/api/datasets/upload")
async def upload_dataset(
    name: str = Form(...),
    description: str = Form(""),
    task_type: str = Form("instance_segmentation"),
    file: UploadFile = File(...)
):
    dest_dir = None
    try:
        # Generate a unique dataset ID
        dataset_id = f"dataset_{int(time.time())}"
        
        # Define extraction target folder
        base_dir = os.path.dirname(os.path.dirname(__file__))
        dest_dir = os.path.join(base_dir, "data", "storage", "raw_datasets", dataset_id)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Read file contents and extract ZIP
        file_bytes = await file.read()
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zip_ref:
            zip_ref.extractall(dest_dir)
            
        # We expect a zip file with folders like "images" and optionally "labels", "metadata".
        # Let's inspect the extracted contents.
        # If there are subdirectories, we make sure we find the "images" directory.
        images_dir = os.path.join(dest_dir, "images")
        
        # If the zip extracted into a subfolder matching the zip name, let's adjust paths
        if not os.path.exists(images_dir):
            # Check if there's a subdirectory containing images
            subdirs = [d for d in os.listdir(dest_dir) if os.path.isdir(os.path.join(dest_dir, d)) and not d.startswith("__")]
            for sd in subdirs:
                sd_images = os.path.join(dest_dir, sd, "images")
                if os.path.exists(sd_images):
                    # Found it! Let's move files up
                    for item in os.listdir(os.path.join(dest_dir, sd)):
                        shutil.move(os.path.join(dest_dir, sd, item), os.path.join(dest_dir, item))
                    break
        
        # Re-check images folder
        images_dir = os.path.join(dest_dir, "images")
        if not os.path.exists(images_dir):
            # If no "images" folder exists at all, let's create a default structure using any found images in the zip!
            os.makedirs(images_dir, exist_ok=True)
            # Find all images at root level of dest_dir
            for item in os.listdir(dest_dir):
                if item.lower().endswith((".png", ".jpg", ".jpeg")) and item != "images":
                    shutil.move(os.path.join(dest_dir, item), os.path.join(images_dir, item))
                    
        # Check metadata and labels directories as well
        labels_dir = os.path.join(dest_dir, "labels")
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir, exist_ok=True)
            for item in os.listdir(dest_dir):
                if item.lower().endswith(".txt") and item != "labels":
                    shutil.move(os.path.join(dest_dir, item), os.path.join(labels_dir, item))
                    
        metadata_dir = os.path.join(dest_dir, "metadata")
        if not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir, exist_ok=True)
            for item in os.listdir(dest_dir):
                if item.lower().endswith(".json") and item != "metadata":
                    shutil.move(os.path.join(dest_dir, item), os.path.join(metadata_dir, item))
        
        # Count images
        img_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        sample_size = len(img_files)
        
        if sample_size == 0:
            raise ValueError("The uploaded ZIP file does not contain any valid images (.png, .jpg, .jpeg) at the root or within an 'images/' folder.")
            
        # Create dataset in DB
        ds = db.create_dataset(
            dataset_id=dataset_id,
            name=name,
            description=description,
            task_type=task_type,
            sample_size=sample_size,
            folder_path=dest_dir
        )
        
        # Automatically generate embeddings and save them
        embedding_service.generate_and_save_embeddings(dataset_id)
        
        return {
            "status": "success",
            "dataset": ds,
            "images_count": sample_size
        }
    except Exception as e:
        # Cleanup directory if extraction or database insert failed
        if dest_dir and os.path.exists(dest_dir):
            shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/datasets/split")
def create_dataset_split(data: DatasetSplitRequest):
    try:
        return dataset_service.create_split_and_export(
            dataset_id=data.dataset_id,
            version_tag=data.version_tag,
            train_ratio=data.train_split,
            val_ratio=data.val_split,
            seed=data.split_seed
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset-versions")
def list_dataset_versions(dataset_id: str = None):
    return db.get_dataset_versions(dataset_id)

@app.get("/api/dataset-versions/{version_id}")
def get_dataset_version(version_id: str):
    version = db.get_dataset_version(version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    return version

@app.get("/api/experiments")
def list_experiments(project_id: str = None):
    return db.get_experiments(project_id)

@app.get("/api/experiments/{exp_id}")
def get_experiment(exp_id: str):
    exp = db.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp

@app.delete("/api/experiments/{exp_id}")
def delete_experiment(exp_id: str):
    exp = db.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    job = db.get_training_job_by_experiment(exp_id)
    if job and job["status"] in ("queued", "preparing_dataset", "training", "evaluating"):
        raise HTTPException(status_code=400, detail="Cannot delete an active training run. Please cancel it first.")
        
    db.delete_experiment(exp_id)
    return {"status": "success", "message": "Experiment run deleted successfully."}

@app.post("/api/experiments")
def create_experiment_and_start_job(data: ExperimentCreate):
    exp_id = f"exp_{int(time.time())}"
    
    # 1. Resolve pipeline_id
    pipeline_id = data.pipeline_id
    if not pipeline_id:
        if data.model_type == "mask_rcnn":
            pipeline_id = "pipe_maskrcnn"
        else:
            pipeline_id = "pipe_yolo"
            
    # 2. Create experiment record linked to pipeline_id
    exp = db.create_experiment(
        exp_id=exp_id,
        project_id=data.project_id,
        dataset_version_id=data.dataset_version_id,
        name=data.name,
        task_type=data.task_type,
        model_type=data.model_type,
        config_dict=data.config,
        pipeline_id=pipeline_id
    )
    
    # 3. Queue the training job
    epochs = data.config.get("epochs", 3)
    job_id = f"job_{int(time.time())}"
    db.create_training_job(job_id, exp_id, total_epochs=epochs)
    
    return {
        "experiment": exp,
        "job_id": job_id
    }

@app.get("/api/experiments/{exp_id}/job")
def get_experiment_training_job(exp_id: str):
    job = db.get_training_job_by_experiment(exp_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found for this experiment")
    
    # Fetch and attach experiment details
    exp = db.get_experiment(exp_id)
    if exp:
        job["experiment"] = exp
        
    return job

@app.get("/api/experiments/{exp_id}/evaluation")
def get_experiment_evaluation(exp_id: str):
    eval_record = db.get_evaluation_by_experiment(exp_id)
    if not eval_record:
        raise HTTPException(status_code=404, detail="Evaluation results not available yet")
    return eval_record

@app.post("/api/experiments/{exp_id}/cancel")
def cancel_experiment_training_job(exp_id: str):
    with db._get_conn() as conn:
        row = conn.execute("SELECT id, status FROM training_jobs WHERE experiment_id = ?", (exp_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Training job not found for this experiment")
        job_id, status = row
        if status not in ("queued", "preparing_dataset", "training"):
            raise HTTPException(status_code=400, detail=f"Job cannot be cancelled in status '{status}'")
            
        # Update statuses to cancelled
        conn.execute("UPDATE training_jobs SET status = 'cancelled' WHERE id = ?", (job_id,))
        conn.execute("UPDATE experiments SET status = 'cancelled' WHERE id = ?", (exp_id,))
        conn.commit()
        
    return {"status": "success", "message": "Job cancellation request sent."}

@app.get("/api/projects/{project_id}/llm-commands")
def get_project_llm_commands(project_id: str):
    return db.get_llm_commands(project_id)

@app.post("/api/projects/{project_id}/llm-commands")
def save_project_llm_command(project_id: str, data: LLMCommandSaveRequest):
    db.save_llm_command(
        project_id=project_id,
        sender=data.sender,
        text=data.text,
        payload=data.payload
    )
    return {"status": "success"}

@app.delete("/api/projects/{project_id}/llm-commands")
def clear_project_llm_commands(project_id: str):
    db.clear_llm_commands(project_id)
    # Re-seed the welcome message
    db.save_llm_command(
        project_id=project_id,
        sender="system",
        text="ATR Assistant terminal online. Ask me to split datasets, configure augmentations, or run training runs."
    )
    return {"status": "success"}

@app.post("/api/llm/command")
def process_llm_command(data: LLMCommandRequest):
    """Processes natural language command, triggers structured workflow backend actions."""
    try:
        # Save user prompt
        db.save_llm_command(
            project_id="proj_default",
            sender="user",
            text=data.prompt
        )
        
        response = llm_service.process_command(data.prompt)
        
        # If successfully parsed into an action, run the action workflow
        if response.get("status") == "success":
            action = response.get("action")
            payload = response.get("payload", {})
            
            if action == "create_experiment":
                # Ensure default project exists
                project_id = "proj_default"
                proj = db.get_project(project_id)
                if not proj:
                    db.create_project(project_id, "Geospatial Target Recognition", "Auto-created project")
                    
                # 1. Export split
                version = dataset_service.create_split_and_export(
                    dataset_id=payload["dataset_id"],
                    version_tag=payload["version_tag"],
                    train_ratio=payload["train_split"],
                    val_ratio=payload["val_split"],
                    seed=payload["split_seed"]
                )
                
                # 2. Create experiment
                exp_id = f"exp_{int(time.time())}"
                
                # Default to CPU training (or Mock mode) in configurations
                config = payload.get("config", {})
                config["mock"] = True  # Default to mock simulator for fast/light UI demonstration unless changed
                
                exp = db.create_experiment(
                    exp_id=exp_id,
                    project_id=project_id,
                    dataset_version_id=version["id"],
                    name=f"LLM Run: {payload['version_tag']}",
                    task_type=payload["task_type"],
                    model_type=payload["model_type"],
                    config_dict=config
                )
                
                # 3. Start training job
                job_id = f"job_{int(time.time())}"
                db.create_training_job(job_id, exp_id, total_epochs=config.get("epochs", 3))
                
                epochs = config.get("epochs", 3)
                batch = config.get("batch", 2)
                imgsz = config.get("imgsz", 512)
                aug = config.get("augmentations", {})
                aug_desc = []
                if aug.get("fliplr"): aug_desc.append("fliplr")
                if aug.get("flipud"): aug_desc.append("flipud")
                if aug.get("degrees", 0.0) > 0.0: aug_desc.append(f"rotate({aug.get('degrees')}deg)")
                aug_text = f", augmentations: {', '.join(aug_desc)}" if aug_desc else ""

                response["workflow_result"] = {
                    "project_id": project_id,
                    "dataset_version_id": version["id"],
                    "experiment_id": exp_id,
                    "job_id": job_id
                }
                response["message"] = (
                    f"Successfully exported '{payload['dataset_id']}' with {int(payload['train_split']*100)}/{int(payload['val_split']*100)} split, "
                    f"created experiment '{exp_id}' (epochs={epochs}, batch={batch}, imgsz={imgsz}{aug_text}), and started training job '{job_id}'."
                )
                
            elif action == "clone_experiment":
                source_id = payload["source_experiment_id"]
                source_exp = db.get_experiment(source_id)
                
                if not source_exp:
                    raise ValueError(f"Source experiment {source_id} not found")
                    
                # Blend old config with changes
                new_config = source_exp["config"].copy()
                
                # Apply changes to augmentations
                new_aug = new_config.get("augmentations", {}).copy()
                incoming_aug = payload.get("changes", {}).get("config", {}).get("augmentations", {})
                new_aug.update(incoming_aug)
                new_config["augmentations"] = new_aug
                
                # Create clone experiment
                exp_id = f"exp_{int(time.time())}"
                exp = db.create_experiment(
                    exp_id=exp_id,
                    project_id=source_exp["project_id"],
                    dataset_version_id=source_exp["dataset_version_id"],
                    name=f"Augment Rerun: {source_exp['name']}",
                    task_type=source_exp["task_type"],
                    model_type=source_exp["model_type"],
                    config_dict=new_config,
                    parent_id=source_id
                )
                
                # Start job
                job_id = f"job_{int(time.time())}"
                db.create_training_job(job_id, exp_id, total_epochs=new_config.get("epochs", 3))
                
                response["workflow_result"] = {
                    "project_id": source_exp["project_id"],
                    "dataset_version_id": source_exp["dataset_version_id"],
                    "experiment_id": exp_id,
                    "job_id": job_id
                }
                response["message"] = f"Successfully cloned experiment '{source_id}', applied augmentations: {new_aug}, and started training job '{job_id}'."
                
            # Save system response with workflow result
            db.save_llm_command(
                project_id="proj_default",
                sender="system",
                text=response.get("message", ""),
                payload=response.get("workflow_result")
            )
        else:
            # Save fallback system response
            db.save_llm_command(
                project_id="proj_default",
                sender="system",
                text=response.get("message", "Processing completed.")
            )
            
        return response
    except ValueError as e:
        err_msg = f"Failed to execute command: {str(e)}"
        db.save_llm_command(
            project_id="proj_default",
            sender="system",
            text=err_msg
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        err_msg = f"Failed to execute command: {str(e)}"
        db.save_llm_command(
            project_id="proj_default",
            sender="system",
            text=err_msg
        )
        raise HTTPException(status_code=500, detail=str(e))

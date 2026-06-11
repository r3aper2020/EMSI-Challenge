#!/usr/bin/env python3
import sys
import os
import json
import time
import traceback

# Ensure app folder is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import Database
from app.services.dataset_service import DatasetService
from app.services.embedding_service import EmbeddingService

def log_debug(msg):
    sys.stderr.write(f"[MCP Debug] {msg}\n")
    sys.stderr.flush()

def call_tool(name, args, db, dataset_service, embedding_service):
    log_debug(f"Calling tool {name} with args {args}")
    if name == "list_datasets":
        return db.get_datasets()
        
    elif name == "curate_dataset_split":
        dataset_id = args["dataset_id"]
        version_tag = args["version_tag"]
        train_split = args.get("train_split", 0.8)
        val_split = args.get("val_split", 0.2)
        split_seed = args.get("split_seed", 42)
        
        return dataset_service.create_split_and_export(
            dataset_id=dataset_id,
            version_tag=version_tag,
            train_ratio=train_split,
            val_ratio=val_split,
            seed=split_seed
        )
        
    elif name == "start_training":
        project_id = args.get("project_id", "proj_default")
        dataset_version_id = args["dataset_version_id"]
        run_name = args["name"]
        task_type = args.get("task_type", "instance_segmentation")
        model_type = args.get("model_type", "yolo_seg")
        
        config = {
            "epochs": args.get("epochs", 3),
            "batch": args.get("batch_size", 2),
            "imgsz": args.get("imgsz", 512),
            "mock": args.get("mock", True),
            "augmentations": {
                "fliplr": args.get("fliplr", False),
                "flipud": args.get("flipud", False),
                "degrees": args.get("degrees", 0.0)
            }
        }
        
        exp_id = f"exp_{int(time.time())}"
        
        if model_type == "mask_rcnn":
            pipeline_id = "pipe_maskrcnn"
        else:
            pipeline_id = "pipe_yolo"
            
        exp = db.create_experiment(
            exp_id=exp_id,
            project_id=project_id,
            dataset_version_id=dataset_version_id,
            name=run_name,
            task_type=task_type,
            model_type=model_type,
            config_dict=config,
            pipeline_id=pipeline_id
        )
        
        job_id = f"job_{int(time.time())}"
        db.create_training_job(job_id, exp_id, total_epochs=config["epochs"])
        
        return {
            "status": "success",
            "experiment_id": exp_id,
            "job_id": job_id,
            "experiment": exp
        }
        
    elif name == "get_training_status":
        exp_id = args["experiment_id"]
        job = db.get_training_job_by_experiment(exp_id)
        if not job:
            return {"error": f"No training job found for experiment '{exp_id}'"}
        return job
        
    elif name == "compare_models":
        base_id = args["base_experiment_id"]
        cand_id = args["candidate_experiment_id"]
        
        base_exp = db.get_experiment(base_id)
        cand_exp = db.get_experiment(cand_id)
        
        if not base_exp or not cand_exp:
            return {"error": "Base or Candidate experiment not found."}
            
        base_eval = db.get_evaluation_by_experiment(base_id)
        cand_eval = db.get_evaluation_by_experiment(cand_id)
        
        return {
            "base": {
                "id": base_id,
                "name": base_exp["name"],
                "config": base_exp["config"],
                "evaluation": base_eval
            },
            "candidate": {
                "id": cand_id,
                "name": cand_exp["name"],
                "config": cand_exp["config"],
                "evaluation": cand_eval
            }
        }
        
    elif name == "get_visual_predictions":
        exp_id = args["experiment_id"]
        eval_record = db.get_evaluation_by_experiment(exp_id)
        if not eval_record:
            return {"error": f"Evaluation predictions not available yet for experiment '{exp_id}'"}
        return {
            "experiment_id": exp_id,
            "dataset_version_id": eval_record["dataset_version_id"],
            "metrics": {
                "mAP50": eval_record["map50"],
                "precision": eval_record["precision"],
                "recall": eval_record["recall"],
                "f1": eval_record["f1"]
            },
            "predictions": eval_record["predictions"]
        }
        
    elif name == "run_auto_label":
        dataset_id = args["dataset_id"]
        image_id = args["image_id"]
        
        ds = db.get_dataset(dataset_id)
        if not ds or not ds.get("folder_path"):
            return {"error": f"Dataset '{dataset_id}' not found"}
            
        labels_dir = os.path.join(ds["folder_path"], "labels")
        label_file = os.path.join(labels_dir, f"{image_id}.txt")
        
        suggestions = []
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
                log_debug(f"Error parsing existing labels: {e}")
                
        import random
        def deterministic_hash(s: str) -> int:
            h = 0
            for char in s:
                h = (h * 31 + ord(char)) & 0xFFFFFFFF
            return h
            
        if existing_labels:
            rnd = random.Random(deterministic_hash(image_id))
            for idx, lbl in enumerate(existing_labels):
                scale = 0.012
                new_poly = []
                for pt in lbl["polygon"]:
                    dx = rnd.uniform(-scale, scale)
                    dy = rnd.uniform(-scale, scale)
                    new_poly.append([max(0.0, min(1.0, pt[0] + dx)), max(0.0, min(1.0, pt[1] + dy))])
                xs = [pt[0] for pt in new_poly]
                ys = [pt[1] for pt in new_poly]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                suggestions.append({
                    "id": f"sugg_{idx}_{int(time.time())}",
                    "class_id": lbl["class_id"],
                    "polygon": new_poly,
                    "bbox": bbox,
                    "confidence": round(rnd.uniform(0.78, 0.96), 2)
                })
        else:
            rnd = random.Random(deterministic_hash(image_id))
            mock_polys = [
                [[0.22, 0.45], [0.28, 0.41], [0.34, 0.45], [0.28, 0.49]],
                [[0.65, 0.22], [0.71, 0.18], [0.77, 0.22], [0.71, 0.26]],
                [[0.45, 0.75], [0.51, 0.71], [0.57, 0.75], [0.51, 0.79]]
            ]
            for idx, poly in enumerate(mock_polys):
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
                    "class_id": 0,
                    "polygon": perturbed_poly,
                    "bbox": bbox,
                    "confidence": round(rnd.uniform(0.74, 0.94), 2)
                })
        return {"status": "success", "suggestions": suggestions}
        
    elif name == "save_annotations":
        dataset_id = args["dataset_id"]
        image_id = args["image_id"]
        labels = args["labels"]
        
        ds = db.get_dataset(dataset_id)
        if not ds or not ds.get("folder_path"):
            return {"error": f"Dataset '{dataset_id}' not found"}
            
        labels_dir = os.path.join(ds["folder_path"], "labels")
        os.makedirs(labels_dir, exist_ok=True)
        label_file = os.path.join(labels_dir, f"{image_id}.txt")
        
        with open(label_file, "w") as f:
            lines = []
            for label in labels:
                class_id = label.get("class_id", 0)
                polygon = label.get("polygon", [])
                if not polygon:
                    continue
                coords_str = " ".join([f"{pt[0]:.5f} {pt[1]:.5f}" for pt in polygon])
                lines.append(f"{class_id} {coords_str}")
            f.write("\n".join(lines))
            
        return {"status": "success", "message": "Annotations saved successfully."}
        
    else:
        raise ValueError(f"Unrecognized tool name '{name}'")

def handle_request(req, db, dataset_service, embedding_service):
    method = req.get("method")
    req_id = req.get("id")
    
    if req_id is None:
        return None
        
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "atr-workbench-mcp",
                    "version": "1.0.0"
                }
            }
        }
        
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "list_datasets",
                        "description": "Lists all geospatial datasets registered in the ATR Model Production Workbench database.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "curate_dataset_split",
                        "description": "Curates a dataset by splitting it into train and validation ratios and exporting the version.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "dataset_id": {"type": "string", "description": "The ID of the registered dataset (e.g. 'dataset_rareplanes_real')."},
                                "version_tag": {"type": "string", "description": "The custom tag version name (e.g. 'run_v1')."},
                                "train_split": {"type": "number", "description": "Ratio of training samples (e.g. 0.8).", "default": 0.8},
                                "val_split": {"type": "number", "description": "Ratio of validation samples (e.g. 0.2).", "default": 0.2},
                                "split_seed": {"type": "integer", "description": "Deterministic seed for shuffling.", "default": 42}
                            },
                            "required": ["dataset_id", "version_tag"]
                        }
                    },
                    {
                        "name": "start_training",
                        "description": "Trigger a model training pipeline run on a curated dataset version split.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "project_id": {"type": "string", "description": "The project ID.", "default": "proj_default"},
                                "dataset_version_id": {"type": "string", "description": "The ID of the dataset version to train on."},
                                "name": {"type": "string", "description": "Descriptive name for the run."},
                                "task_type": {"type": "string", "description": "The task type, e.g. 'instance_segmentation' or 'object_detection'.", "default": "instance_segmentation"},
                                "model_type": {"type": "string", "description": "The model architecture, e.g. 'yolo_seg' or 'mask_rcnn'.", "default": "yolo_seg"},
                                "epochs": {"type": "integer", "description": "Number of training epochs.", "default": 3},
                                "batch_size": {"type": "integer", "description": "Batch size.", "default": 2},
                                "imgsz": {"type": "integer", "description": "Image resolution size in pixels.", "default": 512},
                                "fliplr": {"type": "boolean", "description": "Enable horizontal flip augmentation.", "default": False},
                                "flipud": {"type": "boolean", "description": "Enable vertical flip augmentation.", "default": False},
                                "degrees": {"type": "number", "description": "Rotation angle limit in degrees.", "default": 0.0},
                                "mock": {"type": "boolean", "description": "Run model in simulation mode.", "default": True}
                            },
                            "required": ["dataset_version_id", "name"]
                        }
                    },
                    {
                        "name": "get_training_status",
                        "description": "Poll the progress percent, loss history, validation accuracy (mAP50), and stdout logs of a training job.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "experiment_id": {"type": "string", "description": "The experiment ID (e.g. 'exp_1234')."}
                            },
                            "required": ["experiment_id"]
                        }
                    },
                    {
                        "name": "compare_models",
                        "description": "Compare metrics deltas (mAP50, precision, recall, f1) and hyperparameter configs of a baseline vs candidate model.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "base_experiment_id": {"type": "string", "description": "The baseline experiment ID."},
                                "candidate_experiment_id": {"type": "string", "description": "The candidate experiment ID."}
                            },
                            "required": ["base_experiment_id", "candidate_experiment_id"]
                        }
                    },
                    {
                        "name": "get_visual_predictions",
                        "description": "Retrieve validation image chips prediction overlays (TP/FP/FN types, bounding boxes, confidences) for visual verification.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "experiment_id": {"type": "string", "description": "The experiment ID."}
                            },
                            "required": ["experiment_id"]
                        }
                    },
                    {
                        "name": "run_auto_label",
                        "description": "Query AI active learning object detections/polygons proposals on a target image chip.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "dataset_id": {"type": "string", "description": "The dataset ID (e.g. 'dataset_rareplanes_real')."},
                                "image_id": {"type": "string", "description": "The image ID (e.g. '100_1040010029...')."}
                            },
                            "required": ["dataset_id", "image_id"]
                        }
                    },
                    {
                        "name": "save_annotations",
                        "description": "Commit customized aircraft vector polygon labels back to the local database / labels file.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "dataset_id": {"type": "string", "description": "The dataset ID."},
                                "image_id": {"type": "string", "description": "The image ID."},
                                "labels": {
                                    "type": "array",
                                    "description": "List of annotation objects with class_id, polygon (list of coordinate lists) and bbox.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "class_id": {"type": "integer"},
                                            "polygon": {
                                                "type": "array",
                                                "items": {
                                                    "type": "array",
                                                    "items": {"type": "number"}
                                                }
                                            }
                                        },
                                        "required": ["class_id", "polygon"]
                                    }
                                }
                            },
                            "required": ["dataset_id", "image_id", "labels"]
                        }
                    }
                ]
            }
        }
        
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        try:
            result = call_tool(tool_name, args, db, dataset_service, embedding_service)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32603,
                    "message": f"Error executing tool: {str(e)}",
                    "data": traceback.format_exc()
                }
            }
            
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' not found."
            }
        }

def main():
    db = Database()
    dataset_service = DatasetService(db)
    embedding_service = EmbeddingService(db)
    
    log_debug("MCP Server started listening on stdin.")
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            req_str = line.strip()
            if not req_str:
                continue
                
            request = json.loads(req_str)
            response = handle_request(request, db, dataset_service, embedding_service)
            if response:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:
            log_debug(f"Error in main loop: {traceback.format_exc()}")

if __name__ == "__main__":
    main()

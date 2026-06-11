import os
import re
import json
import requests
import time
import traceback

TOOLS_DECLARATIONS = [
    {
        "name": "list_datasets",
        "description": "Lists all geospatial datasets registered in the ATR Model Production Workbench database.",
        "parameters": {
            "type": "OBJECT",
            "properties": {}
        }
    },
    {
        "name": "curate_dataset_split",
        "description": "Curate and partition a dataset into training and validation splits and export it.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "dataset_id": {"type": "STRING", "description": "The ID of the dataset to split (e.g. 'dataset_rareplanes_real')."},
                "version_tag": {"type": "STRING", "description": "Tag name of the split version (e.g. 'run_v1')."},
                "train_split": {"type": "NUMBER", "description": "Ratio of training samples (e.g. 0.8 for 80%).", "default": 0.8},
                "val_split": {"type": "NUMBER", "description": "Ratio of validation samples (e.g. 0.2 for 20%).", "default": 0.2},
                "split_seed": {"type": "INTEGER", "description": "Deterministic seed for shuffling.", "default": 42}
            },
            "required": ["dataset_id", "version_tag"]
        }
    },
    {
        "name": "list_experiments",
        "description": "Lists all model training experiments and pipelines run history.",
        "parameters": {
            "type": "OBJECT",
            "properties": {}
        }
    },
    {
        "name": "start_training",
        "description": "Spawns a new model training job on a curated dataset version split.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "dataset_version_id": {"type": "STRING", "description": "The ID of the dataset version to train on (e.g. 'version_dataset_rareplanes_real_12345')."},
                "name": {"type": "STRING", "description": "Descriptive name for this model training run."},
                "task_type": {"type": "STRING", "description": "The task, usually 'instance_segmentation' or 'object_detection'.", "default": "instance_segmentation"},
                "model_type": {"type": "STRING", "description": "The model type, e.g. 'yolo_seg' or 'mask_rcnn'.", "default": "yolo_seg"},
                "epochs": {"type": "INTEGER", "description": "Number of epochs to train.", "default": 3},
                "batch_size": {"type": "INTEGER", "description": "Batch size.", "default": 2},
                "imgsz": {"type": "INTEGER", "description": "Image size resolution.", "default": 512},
                "fliplr": {"type": "BOOLEAN", "description": "Enable horizontal flip augmentation.", "default": False},
                "flipud": {"type": "BOOLEAN", "description": "Enable vertical flip augmentation.", "default": False},
                "degrees": {"type": "NUMBER", "description": "Rotation angle limit in degrees.", "default": 0.0},
                "mock": {"type": "BOOLEAN", "description": "Run model training in simulation mode.", "default": True}
            },
            "required": ["dataset_version_id", "name"]
        }
    },
    {
        "name": "get_training_status",
        "description": "Polls active training logs, current epoch, progress percent, loss history and validation accuracy of a model run.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "experiment_id": {"type": "STRING", "description": "The experiment ID (e.g. 'exp_1234')."}
            },
            "required": ["experiment_id"]
        }
    },
    {
        "name": "compare_models",
        "description": "Compares performance metrics (mAP50, precision, recall, f1) and hyperparameter options of a baseline vs candidate model.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "base_experiment_id": {"type": "STRING", "description": "The baseline experiment ID."},
                "candidate_experiment_id": {"type": "STRING", "description": "The candidate experiment ID."}
            },
            "required": ["base_experiment_id", "candidate_experiment_id"]
        }
    },
    {
        "name": "get_visual_predictions",
        "description": "Gets prediction boxes/coordinates details and scores (TP/FP/FN types) for completed runs.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "experiment_id": {"type": "STRING", "description": "The experiment ID."}
            },
            "required": ["experiment_id"]
        }
    },
    {
        "name": "run_auto_label",
        "description": "Trigger active learning auto-label suggestions for a target image chip.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "dataset_id": {"type": "STRING", "description": "The dataset ID (e.g. 'dataset_rareplanes_real')."},
                "image_id": {"type": "STRING", "description": "The target image ID (e.g. '100_1040010029...')."}
            },
            "required": ["dataset_id", "image_id"]
        }
    }
]

def get_dataset_id_from_version(version_id: str) -> str:
    if not version_id:
        return 'dataset_rareplanes_real'
    if version_id.startswith('version_'):
        parts = version_id.split('_')
        return "_".join(parts[1:-1])
    return 'dataset_rareplanes_real'

class LLMService:
    def __init__(self, db):
        self.db = db
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def process_command(self, prompt: str) -> dict:
        print(f"LLM processing prompt: '{prompt}'")
        
        # 1. Try LLM API extraction with function calling if key is present
        if self.api_key:
            try:
                action_data = self._call_gemini_api(prompt)
                if action_data:
                    return self._process_structured_action(action_data, prompt)
            except Exception as e:
                print(f"Gemini API call failed, falling back to rule-based parser: {e}")
                traceback.print_exc()
                
        # 2. Rule-based fallback parser with database stats queries
        action_data = self._fallback_parse(prompt)
        return self._process_structured_action(action_data, prompt)

    def _call_gemini_api(self, prompt: str) -> dict:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        system_instruction = (
            "You are an AI assistant orchestrating geospatial target recognition (ATR) workflows.\n"
            "You have access to tools that query the database, curate splits, run model training, and check status.\n"
            "Translate user requests into tool calls when necessary. Always answer questions fully and conversationally "
            "using the tools data. When triggering curated splits or starting training, let the user know conversationally."
        )
        
        # Build contents with tools definitions
        contents = [
            {
                "role": "user",
                "parts": [{"text": f"System Instruction: {system_instruction}\n\nUser Prompt: {prompt}"}]
            }
        ]
        
        # We will loop to handle function calls (recursively up to 5 times)
        for loop_idx in range(5):
            payload = {
                "contents": contents,
                "tools": [{"functionDeclarations": TOOLS_DECLARATIONS}]
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=12)
            if response.status_code != 200:
                raise Exception(f"Gemini API returned status {response.status_code}: {response.text}")
                
            res_json = response.json()
            candidate = res_json["candidates"][0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            # Append model's response to history
            contents.append(content)
            
            # Check for function calls
            has_fc = False
            tool_responses_parts = []
            
            # Keep track of training/curation results to return to frontend action handler
            action_type = "chat"
            action_payload = {}
            
            for part in parts:
                if "functionCall" in part:
                    has_fc = True
                    fc = part["functionCall"]
                    name = fc["name"]
                    args = fc.get("args", {})
                    
                    # Execute tool locally
                    try:
                        from app.services.dataset_service import DatasetService
                        dataset_service = DatasetService(self.db)
                        
                        if name == "list_datasets":
                            result = self.db.get_datasets()
                        elif name == "curate_dataset_split":
                            dataset_id = args.get("dataset_id")
                            version_tag = args.get("version_tag")
                            train_split = args.get("train_split", 0.8)
                            val_split = args.get("val_split", 0.2)
                            split_seed = args.get("split_seed", 42)
                            
                            # Let the backend handler do it on action return
                            action_type = "create_experiment"
                            action_payload = {
                                "dataset_id": dataset_id,
                                "version_tag": version_tag,
                                "train_split": train_split,
                                "val_split": val_split,
                                "split_seed": split_seed,
                                "task_type": "instance_segmentation",
                                "model_type": "yolo_seg",
                                "config": {"epochs": 3}
                            }
                            result = {"status": "intent_captured", "message": "Creating dataset version split..."}
                        elif name == "list_experiments":
                            result = self.db.get_experiments()
                        elif name == "start_training":
                            # Return creation action to FastAPI backend
                            action_type = "create_experiment"
                            action_payload = {
                                "dataset_id": get_dataset_id_from_version(args.get("dataset_version_id")),
                                "version_tag": args.get("dataset_version_id", "v1"),
                                "train_split": 0.8,
                                "val_split": 0.2,
                                "split_seed": 42,
                                "task_type": args.get("task_type", "instance_segmentation"),
                                "model_type": args.get("model_type", "yolo_seg"),
                                "config": {
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
                            }
                            result = {"status": "intent_captured", "message": "Starting training job..."}
                        elif name == "get_training_status":
                            result = self.db.get_training_job_by_experiment(args.get("experiment_id"))
                        elif name == "compare_models":
                            base_id = args.get("base_experiment_id")
                            cand_id = args.get("candidate_experiment_id")
                            base_exp = self.db.get_experiment(base_id)
                            cand_exp = self.db.get_experiment(cand_id)
                            base_eval = self.db.get_evaluation_by_experiment(base_id)
                            cand_eval = self.db.get_evaluation_by_experiment(cand_id)
                            result = {
                                "base": {"name": base_exp.get("name") if base_exp else "Unknown", "eval": base_eval},
                                "candidate": {"name": cand_exp.get("name") if cand_exp else "Unknown", "eval": cand_eval}
                            }
                        elif name == "get_visual_predictions":
                            result = self.db.get_evaluation_by_experiment(args.get("experiment_id"))
                        elif name == "run_auto_label":
                            result = {"status": "mock", "message": "Proposals generated."}
                        else:
                            result = {"error": f"Tool '{name}' not executed."}
                    except Exception as e:
                        result = {"error": str(e), "traceback": traceback.format_exc()}
                        
                    tool_responses_parts.append({
                        "functionResponse": {
                            "name": name,
                            "response": {"output": result}
                        }
                    })
            
            if not has_fc:
                text = parts[0].get("text", "") if parts else ""
                # If we mapped a training/curation trigger, return it as action redirection
                if action_type != "chat":
                    return {
                        "action": action_type,
                        **action_payload,
                        "custom_message": text
                    }
                return {
                    "action": "chat",
                    "message": text
                }
            
            contents.append({
                "role": "user",
                "parts": tool_responses_parts
            })
            
        return {"action": "chat", "message": "LLM loop timeout."}

    def _fallback_parse(self, prompt: str) -> dict:
        p_lower = prompt.lower()
        
        # 1. Dataset stats queries
        if "how many dataset" in p_lower or "list dataset" in p_lower or "show dataset" in p_lower or "get dataset" in p_lower:
            datasets = self.db.get_datasets()
            if not datasets:
                text = "We currently have no registered datasets in the workbench database."
            else:
                ds_list = [f"• {d['name']} ({d['sample_size']} chips, ID: {d['id']})" for d in datasets]
                text = "Here are the registered datasets in the database:\n" + "\n".join(ds_list)
            return {
                "action": "chat",
                "message": text,
                "payload": {}
            }
            
        # 2. Leaderboard accuracy queries
        if "best model" in p_lower or "best experiment" in p_lower or "accuracy" in p_lower or "leaderboard" in p_lower or "highest score" in p_lower:
            exps = self.db.get_experiments()
            completed = []
            for e in exps:
                if e["status"] == "complete":
                    eval_rec = self.db.get_evaluation_by_experiment(e["id"])
                    map_val = eval_rec["map50"] if eval_rec else 0.72 # Default mock score
                    completed.append((e, map_val))
            
            if not completed:
                text = "No completed training runs were found in the database. Please start a model training job first."
            else:
                completed.sort(key=lambda x: x[1], reverse=True)
                best = completed[0]
                text = f"The best performing model in our run history is '{best[0]['name']}' (ID: {best[0]['id']}) with a mAP50 validation accuracy of {best[1]:.3f}."
            return {
                "action": "chat",
                "message": text,
                "payload": {}
            }

        # 3. Clone previous experiment with augmentations
        if "redo" in p_lower or "clone" in p_lower or "previous experiment" in p_lower:
            fliplr = "horizontal flip" in p_lower or "horizontal-flip" in p_lower or "flip" in p_lower
            flipud = "vertical flip" in p_lower or "vertical-flip" in p_lower
            
            degrees = 0.0
            if "rotation" in p_lower or "rotate" in p_lower:
                degrees = 15.0
                
            return {
                "action": "clone_experiment",
                "source_experiment_id": "previous",
                "changes": {
                    "config": {
                        "augmentations": {
                            "fliplr": fliplr,
                            "flipud": flipud,
                            "degrees": degrees
                        }
                    }
                }
            }
            
        # 4. Standard train/split triggering
        dataset_id = "dataset_rareplanes_real"
        train_split = 0.8
        val_split = 0.2
        split_match = re.search(r"(\d+)/(\d+)", prompt)
        if split_match:
            try:
                t = int(split_match.group(1))
                v = int(split_match.group(2))
                total = t + v
                train_split = t / total
                val_split = v / total
            except:
                pass
                
        task_type = "instance_segmentation"
        model_type = "yolo_seg"
        if "detection" in p_lower or "object detection" in p_lower:
            task_type = "object_detection"
            model_type = "yolo_detect"
            
        version_tag = "v1"
        if "v2" in p_lower:
            version_tag = "v2"
        elif "v3" in p_lower:
            version_tag = "v3"
            
        return {
            "action": "create_experiment",
            "dataset_id": dataset_id,
            "version_tag": version_tag,
            "train_split": train_split,
            "val_split": val_split,
            "split_seed": 42,
            "task_type": task_type,
            "model_type": model_type,
            "config": {
                "epochs": 3,
                "batch": 2,
                "imgsz": 512,
                "augmentations": {
                    "fliplr": False,
                    "flipud": False,
                    "degrees": 0.0
                }
            }
        }

    def _process_structured_action(self, action_data: dict, original_prompt: str) -> dict:
        action = action_data.get("action")
        
        # If it's a conversational chat response, wrap it cleanly
        if action == "chat":
            return {
                "status": "success",
                "action": "chat",
                "message": action_data.get("message", "Processing completed."),
                "payload": action_data.get("payload", {})
            }
            
        if action == "create_experiment":
            dataset_id = action_data.get("dataset_id", "dataset_rareplanes_real")
            version_tag = action_data.get("version_tag", f"v_{str(int(time.time()))[-4:]}")
            train_split = action_data.get("train_split", 0.8)
            val_split = action_data.get("val_split", 0.2)
            seed = action_data.get("split_seed", 42)
            task_type = action_data.get("task_type", "instance_segmentation")
            model_type = action_data.get("model_type", "yolo_seg")
            config = action_data.get("config", {})
            
            # Check if LLM generated a custom response text
            custom_msg = action_data.get("custom_message")
            if custom_msg:
                message = custom_msg
            else:
                message = f"Creating experiment for dataset '{dataset_id}' with {int(train_split*100)}/{int(val_split*100)} split."
                
            return {
                "status": "success",
                "action": "create_experiment",
                "message": message,
                "payload": {
                    "dataset_id": dataset_id,
                    "version_tag": version_tag,
                    "train_split": train_split,
                    "val_split": val_split,
                    "split_seed": seed,
                    "task_type": task_type,
                    "model_type": model_type,
                    "config": config
                }
            }
            
        elif action == "clone_experiment":
            source_ref = action_data.get("source_experiment_id", "previous")
            changes = action_data.get("changes", {})
            
            source_id = None
            if source_ref == "previous":
                exps = self.db.get_experiments()
                if exps:
                    exps = sorted(exps, key=lambda e: e["created_at"], reverse=True)
                    source_id = exps[0]["id"]
            else:
                source_id = source_ref
                
            if not source_id:
                return {
                    "status": "error",
                    "message": "No previous experiment found to clone from."
                }
                
            return {
                "status": "success",
                "action": "clone_experiment",
                "message": f"Cloning and modifying experiment '{source_id}' with new configuration.",
                "payload": {
                    "source_experiment_id": source_id,
                    "changes": changes
                }
            }
            
        elif action == "compare_experiments":
            base_id = action_data.get("base_experiment_id")
            candidate_id = action_data.get("candidate_experiment_id")
            
            return {
                "status": "success",
                "action": "compare_experiments",
                "message": f"Comparing experiment '{base_id}' against '{candidate_id}'.",
                "payload": {
                    "base_experiment_id": base_id,
                    "candidate_experiment_id": candidate_id
                }
            }
            
        else:
            return {
                "status": "error",
                "message": f"Unrecognized LLM action '{action}' parsed from prompt."
            }

import os
import re
import json
import requests
import time

class LLMService:
    def __init__(self, db):
        self.db = db
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def process_command(self, prompt: str) -> dict:
        print(f"LLM processing prompt: '{prompt}'")
        
        # 1. Try LLM API extraction if key is present
        if self.api_key:
            try:
                action_data = self._call_gemini_api(prompt)
                if action_data:
                    return self._process_structured_action(action_data, prompt)
            except Exception as e:
                print(f"Gemini API call failed, falling back to rule-based parser: {e}")
                
        # 2. Rule-based fallback parser
        action_data = self._fallback_parse(prompt)
        return self._process_structured_action(action_data, prompt)

    def _call_gemini_api(self, prompt: str) -> dict:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        system_instruction = (
            "You are an AI assistant orchestrating geospatial target recognition (ATR) workflows.\n"
            "Translate the user's natural language command into a structured JSON action object.\n"
            "Supported actions:\n"
            "1. 'create_experiment'\n"
            "   Params: dataset_id (str, default 'dataset_rareplanes_real'), version_tag (str, default 'v1'), "
            "   train_split (float, default 0.8), val_split (float, default 0.2), split_seed (int, default 42), "
            "   task_type (str, default 'instance_segmentation'), model_type (str, default 'yolo_seg'), "
            "   config (object, containing epochs (int, default 3), batch (int, default 2), imgsz (int, default 512), "
            "   augmentations (object containing fliplr (bool), flipud (bool), degrees (float)))\n"
            "2. 'clone_experiment'\n"
            "   Params: source_experiment_id (str, usually 'previous' or a specific ID), changes (object containing modifications to config)\n"
            "3. 'compare_experiments'\n"
            "   Params: base_experiment_id (str), candidate_experiment_id (str)\n"
            "\n"
            "Output ONLY the JSON object. Do not include markdown code block formatting or explanations."
        )
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"Prompt: {prompt}\n\nSystem Instruction: {system_instruction}"
                }]
            }],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            res_json = response.json()
            text = res_json["candidates"][0]["content"]["parts"][0]["text"]
            # Clean possible markdown wrap
            text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
            text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
            return json.loads(text.strip())
        else:
            raise Exception(f"Gemini API returned status {response.status_code}: {response.text}")

    def _fallback_parse(self, prompt: str) -> dict:
        p_lower = prompt.lower()
        
        # Scenario 1: Clone previous experiment with augmentations
        # e.g., "Redo the previous experiment but apply horizontal flip and random rotation"
        if "redo" in p_lower or "clone" in p_lower or "previous experiment" in p_lower:
            fliplr = "horizontal flip" in p_lower or "horizontal-flip" in p_lower or "flip" in p_lower
            flipud = "vertical flip" in p_lower or "vertical-flip" in p_lower
            
            degrees = 0.0
            if "rotation" in p_lower or "rotate" in p_lower:
                # Default rotation to 15 degrees if unspecified
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
            
        # Determine dataset ID based on keywords
        dataset_id = "dataset_rareplanes_real"
        
        # Try to parse split ratios like 80/20, 70/30
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
                
        # Parse task type
        task_type = "instance_segmentation"
        model_type = "yolo_seg"
        if "detection" in p_lower or "object detection" in p_lower:
            task_type = "object_detection"
            model_type = "yolo_detect"
            
        # Version tag
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
        
        if action == "create_experiment":
            # Extract variables
            dataset_id = action_data.get("dataset_id", "dataset_rareplanes_real")
            version_tag = action_data.get("version_tag", f"v_{str(int(time.time()))[-4:]}")
            train_split = action_data.get("train_split", 0.8)
            val_split = action_data.get("val_split", 0.2)
            seed = action_data.get("split_seed", 42)
            task_type = action_data.get("task_type", "instance_segmentation")
            model_type = action_data.get("model_type", "yolo_seg")
            config = action_data.get("config", {})
            
            return {
                "status": "success",
                "action": "create_experiment",
                "message": f"Creating experiment for dataset '{dataset_id}' with {int(train_split*100)}/{int(val_split*100)} split.",
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
            
            # Find the actual source experiment ID
            source_id = None
            if source_ref == "previous":
                exps = self.db.get_experiments()
                if exps:
                    # Sort by creation time to get the latest
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

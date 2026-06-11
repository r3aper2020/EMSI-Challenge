import os
import sqlite3
import json
import time

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "workbench.db")

class Database:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON;")
            
            # Projects Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at REAL NOT NULL
                )
            """)

            # Pipelines Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipelines (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)
            
            # Datasets Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    task_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    folder_path TEXT NOT NULL
                )
            """)
            
            # Dataset Versions Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    version_tag TEXT NOT NULL,
                    train_split REAL NOT NULL,
                    val_split REAL NOT NULL,
                    split_seed INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    export_path TEXT,
                    created_at REAL NOT NULL,
                    manifest TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
                )
            """)
            
            # Experiments Table (representing Runs of a Pipeline)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    pipeline_id TEXT,
                    dataset_version_id TEXT,
                    name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    parent_experiment_id TEXT,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    run_number INTEGER,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY (pipeline_id) REFERENCES pipelines(id) ON DELETE CASCADE,
                    FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(id)
                )
            """)
            
            # Add pipeline_id and run_number to experiments for existing DBs
            try:
                conn.execute("ALTER TABLE experiments ADD COLUMN pipeline_id TEXT;")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE experiments ADD COLUMN run_number INTEGER;")
            except sqlite3.OperationalError:
                pass
            
            # Training Jobs Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    current_epoch INTEGER DEFAULT 0,
                    total_epochs INTEGER DEFAULT 10,
                    progress_percent REAL DEFAULT 0.0,
                    loss_history_json TEXT,
                    val_loss_history_json TEXT,
                    map50_history_json TEXT,
                    logs TEXT DEFAULT '',
                    metrics_json TEXT,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
                )
            """)
            
            # Evaluations Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL UNIQUE,
                    dataset_version_id TEXT NOT NULL,
                    map50 REAL DEFAULT 0.0,
                    map50_95 REAL DEFAULT 0.0,
                    precision REAL DEFAULT 0.0,
                    recall REAL DEFAULT 0.0,
                    f1 REAL DEFAULT 0.0,
                    predictions_json TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                    FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(id)
                )
            """)
            
            # Embeddings Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
                )
            """)
            
            # Seed default project and pipelines if not present
            try:
                # Check default project
                cursor = conn.execute("SELECT id FROM projects WHERE id = 'proj_default'")
                if not cursor.fetchone():
                    conn.execute(
                        "INSERT INTO projects (id, name, description, created_at) VALUES (?, ?, ?, ?)",
                        ("proj_default", "RarePlanes Aircraft ATR", "Workbench project for managing and training ATR models on airfield datasets.", time.time())
                    )
                
                # Check pipelines
                default_pipelines = [
                    ("pipe_curation", "proj_default", "Dataset Curation", "Outlier/duplicate detection and quality scoring workflows.", "curation"),
                    ("pipe_labeling_review", "proj_default", "Labeling Review", "SME-in-the-loop validation of active learning proposals.", "curation"),
                    ("pipe_yolo", "proj_default", "YOLO Segmentation Training", "Train YOLOv8-seg architectures on geospatial splits.", "segmentation_training"),
                    ("pipe_maskrcnn", "proj_default", "Mask R-CNN Training", "Train heavy Mask R-CNN models for high-fidelity detection.", "segmentation_training"),
                    ("pipe_eval_review", "proj_default", "Evaluation & FiftyOne Review", "mAP comparisons and visual inspection of false negatives.", "evaluation_review")
                ]
                for p_id, proj_id, name, desc, p_type in default_pipelines:
                    cursor = conn.execute("SELECT id FROM pipelines WHERE id = ?", (p_id,))
                    if not cursor.fetchone():
                        conn.execute(
                            "INSERT INTO pipelines (id, project_id, name, description, type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                            (p_id, proj_id, name, desc, p_type, time.time())
                        )
            except Exception as e:
                print(f"Error seeding default projects/pipelines: {e}")
            
            conn.commit()

    # --- Project Methods ---
    def create_project(self, project_id, name, description):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (project_id, name, description, time.time())
            )
            conn.commit()
            return self.get_project(project_id)

    def get_project(self, project_id):
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            return dict(row) if row else None

    def get_projects(self):
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    # --- Dataset Methods ---
    def create_dataset(self, dataset_id, name, description, task_type, sample_size, folder_path):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO datasets (id, name, description, task_type, created_at, sample_size, folder_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (dataset_id, name, description, task_type, time.time(), sample_size, folder_path)
            )
            conn.commit()
            return self.get_dataset(dataset_id)

    def get_dataset(self, dataset_id):
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
            return dict(row) if row else None

    def get_datasets(self):
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    # --- Dataset Version Methods ---
    def create_dataset_version(self, version_id, dataset_id, version_tag, train_split, val_split, split_seed, status, export_path=None, manifest=None):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO dataset_versions (id, dataset_id, version_tag, train_split, val_split, split_seed, status, export_path, created_at, manifest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (version_id, dataset_id, version_tag, train_split, val_split, split_seed, status, export_path, time.time(), json.dumps(manifest) if manifest else None)
            )
            conn.commit()
            return self.get_dataset_version(version_id)

    def get_dataset_version(self, version_id):
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM dataset_versions WHERE id = ?", (version_id,)).fetchone()
            if row:
                d = dict(row)
                d["manifest"] = json.loads(d["manifest"]) if d["manifest"] else None
                return d
            return None

    def get_dataset_versions(self, dataset_id=None):
        with self._get_conn() as conn:
            if dataset_id:
                rows = conn.execute("SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY created_at DESC", (dataset_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM dataset_versions ORDER BY created_at DESC").fetchall()
            res = []
            for r in rows:
                d = dict(r)
                d["manifest"] = json.loads(d["manifest"]) if d["manifest"] else None
                res.append(d)
            return res

    # --- Pipeline Methods ---
    def create_pipeline(self, pipeline_id, project_id, name, description, pipeline_type):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO pipelines (id, project_id, name, description, type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (pipeline_id, project_id, name, description, pipeline_type, time.time())
            )
            conn.commit()
            return self.get_pipeline(pipeline_id)

    def get_pipeline(self, pipeline_id):
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM pipelines WHERE id = ?", (pipeline_id,)).fetchone()
            return dict(row) if row else None

    def get_pipelines(self, project_id=None):
        with self._get_conn() as conn:
            if project_id:
                rows = conn.execute("SELECT * FROM pipelines WHERE project_id = ? ORDER BY created_at ASC", (project_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM pipelines ORDER BY created_at ASC").fetchall()
            return [dict(r) for r in rows]

    # --- Experiment Methods ---
    def create_experiment(self, exp_id, project_id, dataset_version_id, name, task_type, model_type, config_dict, parent_id=None, pipeline_id=None):
        with self._get_conn() as conn:
            if not pipeline_id:
                # Auto-resolve pipeline based on task_type / model_type
                if task_type in ("curation", "dataset_curation") or model_type == "curation":
                    pipeline_id = "pipe_curation"
                elif task_type in ("labeling_review", "labeling") or model_type == "labeling_review":
                    pipeline_id = "pipe_labeling_review"
                elif task_type in ("evaluation_review", "eval_review") or model_type == "evaluation_review":
                    pipeline_id = "pipe_eval_review"
                elif model_type == "mask_rcnn":
                    pipeline_id = "pipe_maskrcnn"
                else:
                    pipeline_id = "pipe_yolo"
                
            # Compute run number
            cursor = conn.execute("SELECT COUNT(*) FROM experiments WHERE pipeline_id = ?", (pipeline_id,))
            run_number = cursor.fetchone()[0] + 1
            
            conn.execute(
                "INSERT INTO experiments (id, project_id, pipeline_id, dataset_version_id, name, task_type, model_type, config_json, parent_experiment_id, status, created_at, run_number) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (exp_id, project_id, pipeline_id, dataset_version_id, name, task_type, model_type, json.dumps(config_dict) if config_dict else "{}", parent_id, "queued", time.time(), run_number)
            )
            conn.commit()
            return self.get_experiment(exp_id)

    def get_experiment(self, exp_id):
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
            if row:
                d = dict(row)
                d["config"] = json.loads(d["config_json"]) if d["config_json"] else {}
                return d
            return None

    def get_experiments(self, project_id=None):
        with self._get_conn() as conn:
            if project_id:
                rows = conn.execute("SELECT * FROM experiments WHERE project_id = ? ORDER BY created_at DESC", (project_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
            res = []
            for r in rows:
                d = dict(r)
                d["config"] = json.loads(d["config_json"]) if d["config_json"] else {}
                res.append(d)
            return res

    def update_experiment_status(self, exp_id, status):
        with self._get_conn() as conn:
            conn.execute("UPDATE experiments SET status = ? WHERE id = ?", (status, exp_id))
            conn.commit()

    def get_project_tree(self, project_id):
        proj = self.get_project(project_id)
        if not proj:
            return None
        
        pipelines = self.get_pipelines(project_id)
        for pipe in pipelines:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE pipeline_id = ? ORDER BY created_at DESC",
                    (pipe["id"],)
                ).fetchall()
            
            pipe_experiments = []
            for r in rows:
                d = dict(r)
                d["config"] = json.loads(d["config_json"]) if d["config_json"] else {}
                
                # Fetch job and evaluation if training
                d["job"] = self.get_training_job_by_experiment(d["id"])
                d["evaluation"] = self.get_evaluation_by_experiment(d["id"])
                
                pipe_experiments.append(d)
                
            pipe["experiments"] = pipe_experiments
            
        proj["pipelines"] = pipelines
        return proj

    # --- Training Job Methods ---
    def create_training_job(self, job_id, experiment_id, total_epochs=10):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO training_jobs (id, experiment_id, status, total_epochs, created_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, experiment_id, "queued", total_epochs, time.time())
            )
            conn.commit()
            return self.get_training_job(job_id)

    def get_training_job(self, job_id):
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
            if row:
                d = dict(row)
                d["loss_history"] = json.loads(d["loss_history_json"]) if d["loss_history_json"] else []
                d["val_loss_history"] = json.loads(d["val_loss_history_json"]) if d["val_loss_history_json"] else []
                d["map50_history"] = json.loads(d["map50_history_json"]) if d["map50_history_json"] else []
                d["metrics"] = json.loads(d["metrics_json"]) if d["metrics_json"] else {}
                return d
            return None

    def get_training_job_by_experiment(self, exp_id):
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM training_jobs WHERE experiment_id = ?", (exp_id,)).fetchone()
            if row:
                d = dict(row)
                d["loss_history"] = json.loads(d["loss_history_json"]) if d["loss_history_json"] else []
                d["val_loss_history"] = json.loads(d["val_loss_history_json"]) if d["val_loss_history_json"] else []
                d["map50_history"] = json.loads(d["map50_history_json"]) if d["map50_history_json"] else []
                d["metrics"] = json.loads(d["metrics_json"]) if d["metrics_json"] else {}
                return d
            return None

    def get_queued_training_jobs(self):
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM training_jobs WHERE status = 'queued' ORDER BY created_at ASC").fetchall()
            return [dict(r) for r in rows]

    def update_training_job(self, job_id, status, current_epoch=None, progress_percent=None, 
                            loss_history=None, val_loss_history=None, map50_history=None, logs=None, metrics=None):
        with self._get_conn() as conn:
            updates = []
            params = []
            
            updates.append("status = ?")
            params.append(status)
            
            if current_epoch is not None:
                updates.append("current_epoch = ?")
                params.append(current_epoch)
            if progress_percent is not None:
                updates.append("progress_percent = ?")
                params.append(progress_percent)
            if loss_history is not None:
                updates.append("loss_history_json = ?")
                params.append(json.dumps(loss_history))
            if val_loss_history is not None:
                updates.append("val_loss_history_json = ?")
                params.append(json.dumps(val_loss_history))
            if map50_history is not None:
                updates.append("map50_history_json = ?")
                params.append(json.dumps(map50_history))
            if logs is not None:
                # Append to existing logs
                row = conn.execute("SELECT logs FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
                existing = row[0] if row and row[0] else ""
                new_logs = existing + "\n" + logs if existing else logs
                updates.append("logs = ?")
                params.append(new_logs)
            if metrics is not None:
                updates.append("metrics_json = ?")
                params.append(json.dumps(metrics))
                
            if status in ("complete", "failed", "cancelled"):
                updates.append("completed_at = ?")
                params.append(time.time())
                
            params.append(job_id)
            query = f"UPDATE training_jobs SET {', '.join(updates)} WHERE id = ?"
            conn.execute(query, params)
            conn.commit()

    # --- Evaluation Methods ---
    def create_evaluation(self, eval_id, experiment_id, dataset_version_id, map50, map50_95, precision, recall, f1, predictions=None):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO evaluations (id, experiment_id, dataset_version_id, map50, map50_95, precision, recall, f1, predictions_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (eval_id, experiment_id, dataset_version_id, map50, map50_95, precision, recall, f1, json.dumps(predictions) if predictions else None, time.time())
            )
            conn.commit()
            return self.get_evaluation(eval_id)

    def get_evaluation(self, eval_id):
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM evaluations WHERE id = ?", (eval_id,)).fetchone()
            if row:
                d = dict(row)
                d["predictions"] = json.loads(d["predictions_json"]) if d["predictions_json"] else None
                return d
            return None

    def get_evaluation_by_experiment(self, exp_id):
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM evaluations WHERE experiment_id = ?", (exp_id,)).fetchone()
            if row:
                d = dict(row)
                d["predictions"] = json.loads(d["predictions_json"]) if d["predictions_json"] else None
                return d
            return None

    # --- Embeddings Methods ---
    def save_embeddings(self, dataset_id, image_id, x, y, metadata_dict):
        with self._get_conn() as conn:
            emb_id = f"emb_{dataset_id}_{image_id}"
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (id, dataset_id, image_id, x, y, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                (emb_id, dataset_id, image_id, x, y, json.dumps(metadata_dict))
            )
            conn.commit()

    def get_embeddings_by_dataset(self, dataset_id):
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM embeddings WHERE dataset_id = ?", (dataset_id,)).fetchall()
            res = []
            for r in rows:
                d = dict(r)
                d["metadata"] = json.loads(d["metadata_json"]) if d["metadata_json"] else {}
                res.append(d)
            return res

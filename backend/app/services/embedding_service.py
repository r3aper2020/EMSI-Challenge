import os
import json
import time
import numpy as np
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

class EmbeddingService:
    def __init__(self, db):
        self.db = db
        # Use CPU for feature extraction as it is fast enough for 60 images
        self.device = torch.device("cpu")
        
        # Load pre-trained ResNet18 model as feature extractor
        try:
            # Modern torchvision syntax
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        except ImportError:
            # Legacy torchvision syntax
            self.model = models.resnet18(pretrained=True)
            
        # Strip final classification fc layer, replace with Identity
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms matching ImageNet training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def generate_and_save_embeddings(self, dataset_id):
        dataset = self.db.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        source_dir = dataset["folder_path"]
        images_dir = os.path.join(source_dir, "images")
        metadata_dir = os.path.join(source_dir, "metadata")
        
        if not os.path.exists(images_dir):
            return []
            
        img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if not img_files:
            return []
            
        # Limit embeddings to 300 for visualization performance (prevents blocking startup / frontend lag)
        max_embs = 300
        if len(img_files) > max_embs:
            import random
            random.seed(42)
            img_files = random.sample(img_files, max_embs)
            img_files = sorted(img_files)
            
        print(f"Extracting ResNet18 embeddings for {len(img_files)} images...")
        embeddings = []
        img_ids = []
        meta_list = []
        
        # 1. Extract raw 512-dimensional embeddings
        with torch.no_grad():
            for f in img_files:
                img_path = os.path.join(images_dir, f)
                img_id = os.path.splitext(f)[0]
                
                # Load image
                image = Image.open(img_path).convert("RGB")
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # Extract feature vector
                feat = self.model(input_tensor).squeeze().cpu().numpy()
                embeddings.append(feat)
                img_ids.append(img_id)
                
                # Load metadata
                meta_path = os.path.join(metadata_dir, f"{img_id}.json")
                meta_data = {}
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as mf:
                        meta_data = json.load(mf)
                meta_list.append(meta_data)
                
        embeddings = np.array(embeddings) # shape: (N, 512)
        
        # 2. Perform PCA in NumPy to project to 2D
        print("Running PCA projection...")
        N = len(embeddings)
        if N > 2:
            # Center data
            mean_vec = np.mean(embeddings, axis=0)
            X_centered = embeddings - mean_vec
            
            # Covariance matrix
            cov = np.cov(X_centered, rowvar=False)
            
            # Eigenvalues & Eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Project onto top 2 components
            X_2d = np.dot(X_centered, eigenvectors[:, :2])
            
            # Normalize projection output to fit nicely in a [-10, 10] box
            x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
            y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
            
            if x_max > x_min:
                X_2d[:, 0] = ((X_2d[:, 0] - x_min) / (x_max - x_min)) * 20 - 10
            if y_max > y_min:
                X_2d[:, 1] = ((X_2d[:, 1] - y_min) / (y_max - y_min)) * 20 - 10
        else:
            # Fallback for tiny sizes
            X_2d = np.random.uniform(-5, 5, size=(N, 2))
            
        # 3. Save to database
        for i, img_id in enumerate(img_ids):
            x = float(X_2d[i, 0])
            y = float(X_2d[i, 1])
            self.db.save_embeddings(
                dataset_id=dataset_id,
                image_id=img_id,
                x=x,
                y=y,
                metadata_dict=meta_list[i]
            )
            
        print(f"Embeddings saved successfully for {N} images.")
        return self.db.get_embeddings_by_dataset(dataset_id)

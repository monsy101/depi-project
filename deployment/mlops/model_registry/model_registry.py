#!/usr/bin/env python3
"""
Model Registry for Stable Diffusion Models
Manages model versioning, metadata, and deployment readiness
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Any
import yaml

class ModelRegistry:
    """Model registry for tracking and managing Stable Diffusion models"""

    def __init__(self, registry_dir="./model_registry"):
        """
        Initialize model registry

        Args:
            registry_dir: Directory to store model registry data
        """
        self.registry_dir = Path(registry_dir)
        self.models_dir = self.registry_dir / "models"
        self.metadata_dir = self.registry_dir / "metadata"
        self.registry_file = self.registry_dir / "registry.json"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Load or create registry
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load existing registry or create new one"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "models": {},
                "versions": {},
                "deployments": {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }

    def _save_registry(self):
        """Save registry to disk"""
        self.registry["updated_at"] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register_model(self, model_path: str, model_type: str, metadata: Dict[str, Any] = None) -> str:
        """
        Register a new model in the registry

        Args:
            model_path: Path to the model files
            model_type: Type of model (base, lora_adapter, etc.)
            metadata: Additional metadata for the model

        Returns:
            model_id: Unique identifier for the registered model
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")

        # Generate model ID based on content hash
        model_hash = self._calculate_model_hash(model_path)
        model_id = f"{model_type}_{model_hash[:8]}"

        # Create model entry
        model_entry = {
            "model_id": model_id,
            "model_type": model_type,
            "original_path": str(model_path),
            "registered_at": datetime.now().isoformat(),
            "file_size": self._calculate_directory_size(model_path),
            "metadata": metadata or {},
            "versions": []
        }

        # Copy model to registry
        registry_model_path = self.models_dir / model_id
        if registry_model_path.exists():
            shutil.rmtree(registry_model_path)
        shutil.copytree(model_path, registry_model_path)

        # Save metadata
        metadata_file = self.metadata_dir / f"{model_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_entry, f, indent=2, default=str)

        # Update registry
        self.registry["models"][model_id] = model_entry
        self._save_registry()

        print(f"Model {model_id} registered successfully")
        return model_id

    def create_version(self, model_id: str, version_name: str, description: str = "",
                      metrics: Dict[str, Any] = None) -> str:
        """
        Create a new version for an existing model

        Args:
            model_id: ID of the model to version
            version_name: Name for this version
            description: Description of the version
            metrics: Performance metrics for this version

        Returns:
            version_id: Unique identifier for the version
        """
        if model_id not in self.registry["models"]:
            raise ValueError(f"Model {model_id} not found in registry")

        version_id = f"{model_id}_v{len(self.registry['models'][model_id]['versions']) + 1}"

        version_entry = {
            "version_id": version_id,
            "model_id": model_id,
            "version_name": version_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "deployment_ready": False,
            "tags": []
        }

        # Add version to model's versions list
        self.registry["models"][model_id]["versions"].append(version_id)

        # Add to versions registry
        self.registry["versions"][version_id] = version_entry

        self._save_registry()

        print(f"Version {version_id} created for model {model_id}")
        return version_id

    def mark_deployment_ready(self, version_id: str, deployment_config: Dict[str, Any] = None):
        """
        Mark a model version as ready for deployment

        Args:
            version_id: ID of the version to mark as deployment-ready
            deployment_config: Configuration for deployment
        """
        if version_id not in self.registry["versions"]:
            raise ValueError(f"Version {version_id} not found")

        self.registry["versions"][version_id]["deployment_ready"] = True
        self.registry["versions"][version_id]["deployment_config"] = deployment_config or {}
        self.registry["versions"][version_id]["deployment_ready_at"] = datetime.now().isoformat()

        self._save_registry()
        print(f"Version {version_id} marked as deployment-ready")

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a registered model"""
        if model_id not in self.registry["models"]:
            raise ValueError(f"Model {model_id} not found")
        return self.registry["models"][model_id]

    def get_version_info(self, version_id: str) -> Dict[str, Any]:
        """Get information about a model version"""
        if version_id not in self.registry["versions"]:
            raise ValueError(f"Version {version_id} not found")
        return self.registry["versions"][version_id]

    def list_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """List all registered models, optionally filtered by type"""
        models = list(self.registry["models"].values())

        if model_type:
            models = [m for m in models if m["model_type"] == model_type]

        return models

    def list_deployment_ready_versions(self) -> List[Dict[str, Any]]:
        """List all versions that are ready for deployment"""
        return [v for v in self.registry["versions"].values() if v.get("deployment_ready", False)]

    def get_model_path(self, model_id: str) -> Path:
        """Get the path to a registered model's files"""
        return self.models_dir / model_id

    def export_registry_summary(self, output_file: str = "registry_summary.yaml"):
        """Export a summary of the registry to YAML"""
        summary = {
            "total_models": len(self.registry["models"]),
            "total_versions": len(self.registry["versions"]),
            "deployment_ready_versions": len(self.list_deployment_ready_versions()),
            "models_by_type": {},
            "recent_registrations": []
        }

        # Count models by type
        for model in self.registry["models"].values():
            model_type = model["model_type"]
            summary["models_by_type"][model_type] = summary["models_by_type"].get(model_type, 0) + 1

        # Get recent registrations (last 5)
        sorted_models = sorted(
            self.registry["models"].values(),
            key=lambda x: x["registered_at"],
            reverse=True
        )[:5]
        summary["recent_registrations"] = [
            {
                "model_id": m["model_id"],
                "type": m["model_type"],
                "registered": m["registered_at"]
            } for m in sorted_models
        ]

        with open(output_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

        print(f"Registry summary exported to {output_file}")

    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate hash of model files for versioning"""
        hash_md5 = hashlib.md5()

        if model_path.is_file():
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            # For directories, hash all files
            for file_path in sorted(model_path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def _calculate_directory_size(self, path: Path) -> int:
        """Calculate total size of directory in bytes"""
        total_size = 0
        if path.is_file():
            return path.stat().st_size

        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size

def initialize_model_registry():
    """Initialize and return a model registry instance"""
    return ModelRegistry()

if __name__ == "__main__":
    # Example usage
    registry = initialize_model_registry()

    # Export registry summary
    registry.export_registry_summary()

    print("Model registry initialized!")
    print(f"Registry location: {registry.registry_dir}")
    print(f"Models directory: {registry.models_dir}")
    print(f"Metadata directory: {registry.metadata_dir}")


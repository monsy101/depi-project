#!/usr/bin/env python3
"""
MLflow Tracking for Stable Diffusion Model Training
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import os
from datetime import datetime
import json

class MLflowTracker:
    """MLflow tracking wrapper for Stable Diffusion experiments"""

    def __init__(self, experiment_name="stable-diffusion-finetuning", tracking_uri="./mlruns"):
        """
        Initialize MLflow tracking

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        self.client = MlflowClient(tracking_uri)

    def start_run(self, run_name=None):
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        return mlflow.active_run().info.run_id

    def log_params(self, params):
        """Log training parameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics, step=None):
        """Log training metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(self, model, model_name="stable_diffusion_model"):
        """Log PyTorch model"""
        mlflow.pytorch.log_model(model, model_name)

    def log_lora_adapter(self, adapter_path, model_name="lora_adapter"):
        """Log LoRA adapter files"""
        if os.path.exists(adapter_path):
            mlflow.log_artifacts(adapter_path, model_name)

    def log_config(self, config_dict, filename="training_config.json"):
        """Log training configuration"""
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        mlflow.log_artifact(filename)

    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()

    def get_best_run(self, metric_name="loss", mode="min"):
        """Get the best run based on a metric"""
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])

        if mode == "min":
            best_run = runs.loc[runs[metric_name].idxmin()]
        else:
            best_run = runs.loc[runs[metric_name].idxmax()]

        return best_run

def setup_mlflow_tracking():
    """Initialize MLflow tracking for the project"""
    tracker = MLflowTracker()

    # Log system information
    import torch
    system_info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
    }

    tracker.start_run("system_setup")
    tracker.log_params(system_info)
    tracker.end_run()

    return tracker

def log_training_metrics(tracker, epoch, loss, learning_rate, validation_metrics=None):
    """Log training metrics during training"""
    metrics = {
        "epoch": epoch,
        "loss": loss,
        "learning_rate": learning_rate
    }

    if validation_metrics:
        metrics.update(validation_metrics)

    tracker.log_metrics(metrics, step=epoch)

if __name__ == "__main__":
    # Example usage
    tracker = setup_mlflow_tracking()

    # Example training run
    tracker.start_run("example_training")

    # Log parameters
    params = {
        "learning_rate": 1e-5,
        "batch_size": 4,
        "num_epochs": 100,
        "model": "stable-diffusion-v1-5",
        "lora_rank": 16
    }
    tracker.log_params(params)

    # Simulate training metrics
    for epoch in range(10):
        loss = 0.5 * (0.95 ** epoch)  # Simulated decreasing loss
        log_training_metrics(tracker, epoch, loss, params["learning_rate"])

    tracker.end_run()
    print("MLflow tracking setup complete!")


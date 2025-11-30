#!/usr/bin/env python3
"""
TensorBoard Monitoring for Stable Diffusion Training
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import io

class TensorBoardMonitor:
    """TensorBoard monitoring wrapper for Stable Diffusion training"""

    def __init__(self, log_dir="./tensorboard_logs"):
        """
        Initialize TensorBoard writer

        Args:
            log_dir: Directory to store TensorBoard logs
        """
        self.log_dir = log_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"run_{timestamp}")

        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(self.run_dir)

        print(f"TensorBoard logs will be saved to: {self.run_dir}")

    def log_scalars(self, tag, value, step):
        """Log scalar values (loss, learning rate, etc.)"""
        self.writer.add_scalar(tag, value, step)

    def log_training_metrics(self, epoch, loss, learning_rate, grad_norm=None):
        """Log common training metrics"""
        self.log_scalars("Loss/train", loss, epoch)
        self.log_scalars("Learning_Rate", learning_rate, epoch)

        if grad_norm is not None:
            self.log_scalars("Gradient_Norm", grad_norm, epoch)

    def log_validation_metrics(self, epoch, metrics):
        """Log validation metrics"""
        for key, value in metrics.items():
            self.log_scalars(f"Validation/{key}", value, epoch)

    def log_generated_images(self, epoch, images, prompts=None, tag="generated_images"):
        """Log generated images to TensorBoard"""
        if isinstance(images, list):
            for i, img in enumerate(images):
                img_tag = f"{tag}_{i}"
                self._log_single_image(img_tag, img, epoch, prompt=prompts[i] if prompts else None)
        else:
            self._log_single_image(tag, images, epoch, prompt=prompts[0] if prompts else None)

    def _log_single_image(self, tag, image, step, prompt=None):
        """Log a single image with optional prompt as text"""
        if isinstance(image, str):
            # Assume it's a file path
            img = Image.open(image)
        elif isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            img = self._tensor_to_image(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image  # Assume it's already a PIL Image

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Add image to TensorBoard
        self.writer.add_image(tag, np.array(img).transpose(2, 0, 1), step, dataformats='CHW')

        # Add prompt as text if provided
        if prompt:
            self.writer.add_text(f"{tag}_prompt", prompt, step)

    def _tensor_to_image(self, tensor):
        """Convert PyTorch tensor to PIL Image"""
        # Assume tensor is in CHW format, normalize to [0, 255]
        if tensor.dim() == 4:  # Batch dimension
            tensor = tensor[0]  # Take first image

        # Denormalize if needed (assuming ImageNet normalization)
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]

        # Convert to numpy and transpose to HWC
        img_array = tensor.detach().cpu().numpy().transpose(1, 2, 0)

        # Clip to [0, 1] and convert to uint8
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def log_model_graph(self, model, input_size=(1, 3, 512, 512)):
        """Log model architecture graph"""
        try:
            dummy_input = torch.randn(input_size)
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"Could not log model graph: {e}")

    def log_histogram(self, tag, values, step, bins=100):
        """Log histogram of values (useful for weights, activations)"""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        self.writer.add_histogram(tag, values, step, bins=bins)

    def log_pr_curves(self, tag, labels, predictions, step):
        """Log precision-recall curves"""
        self.writer.add_pr_curve(tag, labels, predictions, step)

    def log_embeddings(self, embeddings, metadata=None, label_img=None, tag="embeddings", step=0):
        """Log embeddings for visualization"""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        self.writer.add_embedding(
            embeddings,
            metadata=metadata,
            label_img=label_img,
            global_step=step,
            tag=tag
        )

    def flush(self):
        """Flush all pending writes"""
        self.writer.flush()

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()

def create_monitoring_dashboard(log_dir="./tensorboard_logs"):
    """Create and return a TensorBoard monitor instance"""
    return TensorBoardMonitor(log_dir)

def log_training_progress(monitor, epoch, loss, lr, validation_loss=None, generated_image=None):
    """Convenience function to log common training progress"""
    monitor.log_training_metrics(epoch, loss, lr)

    if validation_loss is not None:
        monitor.log_scalars("Loss/validation", validation_loss, epoch)

    if generated_image is not None:
        monitor.log_generated_images(epoch, generated_image, tag="sample_generation")

if __name__ == "__main__":
    # Example usage
    monitor = create_monitoring_dashboard()

    # Simulate training progress
    print("Logging example training progress to TensorBoard...")

    for epoch in range(10):
        # Simulate training metrics
        loss = 0.5 * (0.9 ** epoch)
        lr = 1e-5 * (0.95 ** epoch)

        log_training_progress(monitor, epoch, loss, lr)

        # Log some additional metrics
        monitor.log_scalars("Custom/Memory_Usage", np.random.uniform(0.5, 0.9), epoch)
        monitor.log_scalars("Custom/GPU_Utilization", np.random.uniform(0.3, 0.95), epoch)

    monitor.close()
    print(f"TensorBoard monitoring complete! View logs with: tensorboard --logdir {monitor.log_dir}")


#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Configuration flag: 
# 0 = do fine-tuning and then testing
# 1 = load saved fine-tuned model and run testing only
USE_PRETRAINED_FINETUNED = 1  # Change this value to control the workflow
# Add this at the top of your code with other global flags
CLASS_IMBALANCE = 1  # Set to 1 to enable class weight balancing, 0 for original approach
# Add this at the top of your code with other global flags
BALANCED_SAMPLING = 1  # Set to 1 to enable balanced batch sampling, 0 for original approach
# Add this at the top of your code with other global flags
SAVE_HEATMAPS_FEATURES = 1  # Set to 1 to save heatmaps and features, 0 to ignore
import numpy as np
import torch
import os
import time
from tqdm import tqdm
import glob
import pickle
from PIL import Image
import torchvision.transforms as transforms
import warnings
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

# Suppress specific warnings from torchvision
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms._transforms_video")

# SlowFast imports
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.utils.parser import parse_args
from slowfast.config.defaults import get_cfg
from slowfast.utils.misc import launch_job
from models import build_model

logger = logging.get_logger(__name__)

def load_bboxes(bbox_dir, video_id):
    """
    Load bounding box data for a video segment.
    """
    bbox_file = os.path.join(bbox_dir, f"{video_id}_bboxes.pkl")
    if not os.path.exists(bbox_file):
        logger.warning(f"No bounding box file found for {video_id} at {bbox_file}")
        return None

    try:
        with open(bbox_file, 'rb') as f:
            bboxes = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading bounding box file for {video_id}: {e}")
        return None

    if not isinstance(bboxes, list) or not bboxes:
        logger.error(f"Invalid bounding box data for {video_id}: expected a non-empty list, got {type(bboxes)}")
        return None

    for i, box in enumerate(bboxes):
        if not isinstance(box, (list, np.ndarray)) or len(box) != 4:
            logger.error(f"Invalid bounding box at frame {i} for {video_id}: expected 4 values, got {box}")
            return None

    return bboxes


class SinglePickleFrameDataset(torch.utils.data.Dataset):
    def __init__(self, frames, video_id, label, cfg, bbox_dir, is_train=True):
        self.frames = frames
        self.video_id = video_id
        self.label = label
        self.cfg = cfg
        self.bbox_dir = bbox_dir
        self.is_train = is_train
        self.bboxes = load_bboxes(bbox_dir, video_id)

        # Use more augmentations for training
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            ])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        num_frames_fast = self.cfg.DATA.NUM_FRAMES
        alpha = self.cfg.SLOWFAST.ALPHA
        num_frames_slow = num_frames_fast // alpha

        # Sample indices
        if len(self.frames) < num_frames_fast:
            fast_indices = list(range(len(self.frames))) + [len(self.frames)-1] * (num_frames_fast - len(self.frames))
        else:
            step = max(1, len(self.frames)//num_frames_fast)
            fast_indices = [i for i in range(0, len(self.frames), step)][:num_frames_fast]

        if len(fast_indices) < num_frames_fast:
            # Pad with duplicates of last frame
            fast_indices += [fast_indices[-1]] * (num_frames_fast - len(fast_indices))

        if len(self.frames) < num_frames_slow:
            slow_indices = list(range(len(self.frames))) + [len(self.frames)-1] * (num_frames_slow - len(self.frames))
        else:
            step = max(1, len(self.frames)//num_frames_slow)
            slow_indices = [i for i in range(0, len(self.frames), step)][:num_frames_slow]
            
        if len(slow_indices) < num_frames_slow:
            # Pad with duplicates of last frame
            slow_indices += [slow_indices[-1]] * (num_frames_slow - len(slow_indices))

        def crop_frame(frame, bbox):
            if bbox is None or len(bbox) != 4:
                return frame
            x_min, y_min, x_max, y_max = map(int, bbox)
            h, w = frame.shape[:2]
            # Add some padding around the bbox
            y_max_ext = min(h, y_max + 30)
            x_min, x_max = max(0, x_min - 10), min(w, x_max + 10)
            y_min = max(0, y_min - 10)
            if x_max <= x_min or y_max_ext <= y_min:
                return frame
            return frame[y_min:y_max_ext, x_min:x_max]

        fast_processed, fast_original = [], []
        slow_processed, slow_original = [], []

        # Fast
        for i in fast_indices:
            try:
                frame = self.frames[i]
                if isinstance(frame, torch.Tensor):
                    frame = frame.permute(1,2,0).numpy()
                elif isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0]==1:
                    frame = frame.squeeze(0)
                if frame.dtype != np.uint8:
                    frame = (frame*255).astype(np.uint8) if frame.max()<=1.0 else frame.astype(np.uint8)
                if frame.ndim!=3 or frame.shape[2]!=3:
                    frame = np.zeros((224,224,3),dtype=np.uint8)
                bbox = (self.bboxes[i] if self.bboxes and i<len(self.bboxes) else None)
                cropped = crop_frame(frame, bbox)
                fast_original.append(cropped)
                fast_processed.append(self.transform(Image.fromarray(cropped)))
            except Exception as e:
                logger.warning(f"Error processing fast frame {i}: {e}")
                # Create a blank frame if processing fails
                fast_original.append(np.zeros((224,224,3), dtype=np.uint8))
                fast_processed.append(torch.zeros((3, 224, 224)))

        # Slow
        for i in slow_indices:
            try:
                frame = self.frames[i]
                if isinstance(frame, torch.Tensor):
                    frame = frame.permute(1,2,0).numpy()
                elif isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0]==1:
                    frame = frame.squeeze(0)
                if frame.dtype != np.uint8:
                    frame = (frame*255).astype(np.uint8) if frame.max()<=1.0 else frame.astype(np.uint8)
                if frame.ndim!=3 or frame.shape[2]!=3:
                    frame = np.zeros((224,224,3),dtype=np.uint8)
                bbox = (self.bboxes[i] if self.bboxes and i<len(self.bboxes) else None)
                cropped = crop_frame(frame, bbox)
                slow_original.append(cropped)
                slow_processed.append(self.transform(Image.fromarray(cropped)))
            except Exception as e:
                logger.warning(f"Error processing slow frame {i}: {e}")
                # Create a blank frame if processing fails
                slow_original.append(np.zeros((224,224,3), dtype=np.uint8))
                slow_processed.append(torch.zeros((3, 224, 224)))

        # Stack tensors
        if fast_processed and slow_processed:
            fast_tensor = torch.stack(fast_processed, dim=0).permute(1,0,2,3)
            slow_tensor = torch.stack(slow_processed, dim=0).permute(1,0,2,3)
        else:
            fast_tensor = torch.zeros((3, num_frames_fast, 224,224))
            slow_tensor = torch.zeros((3, num_frames_slow,224,224))
            fast_original = [np.zeros((224,224,3),dtype=np.uint8)]*num_frames_fast
            slow_original = [np.zeros((224,224,3),dtype=np.uint8)]*num_frames_slow

        return [slow_tensor, fast_tensor], self.video_id, slow_original, fast_original, torch.tensor(self.label, dtype=torch.long)


def fastslow_collate_fn(batch):
    """
    Custom collate_fn: stacks only the Slow/Fast tensors & labels.
    Leaves video_ids and raw-frames as Python lists.
    """
    slow_batch = torch.stack([item[0][0] for item in batch], dim=0)
    fast_batch = torch.stack([item[0][1] for item in batch], dim=0)
    inputs = [slow_batch, fast_batch]

    video_ids     = [item[1] for item in batch]
    slow_raw      = [item[2] for item in batch]
    fast_raw      = [item[3] for item in batch]
    labels        = torch.stack([item[4] for item in batch], dim=0)

    return inputs, video_ids, slow_raw, fast_raw, labels


def modify_slowfast_head(model, num_classes, device):
    in_features = model.head.projection.in_features
    model.head.projection = torch.nn.Linear(in_features, num_classes).to(device)
    logger.info(f"Modified SlowFast head to {num_classes} classes on {device}")
    return model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture activations and gradients
        self.hook_handles = []
        self.hook_handles.append(target_layer.register_forward_hook(self.save_activations))
        self.hook_handles.append(target_layer.register_backward_hook(self.save_gradients))

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, inputs, target_class=None):
        """
        Compute Grad-CAM heatmap for the target class.
        Args:
            inputs: Input tensors [slow_tensor, fast_tensor].
            target_class: Target class index for Grad-CAM. If None, use the predicted class.
        Returns:
            heatmap: Grad-CAM heatmap of shape (T, H, W).
        """
        # Enable gradients for inputs
        inputs_with_grad = []
        for inp in inputs:
            inp_clone = inp.clone().detach().requires_grad_(True)
            inputs_with_grad.append(inp_clone)
        
        self.model.eval()
        self.model.zero_grad()

        # Forward pass with gradients enabled
        outputs, _ = self.model(inputs_with_grad)
        
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        # Create a one-hot target tensor
        target = torch.zeros_like(outputs)
        target[0, target_class] = 1

        # Backward pass to get gradients
        outputs.backward(gradient=target)

        # Compute Grad-CAM - use only the activations and gradients
        if self.gradients is None or self.activations is None:
            # If gradients or activations weren't captured, return dummy heatmap
            logger.warning("Gradients or activations weren't captured for Grad-CAM")
            return np.zeros((8, 7, 7))  # Dummy shape for a typical heatmap

        # Process the gradients and activations
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # Average over spatial dims
        heatmap = torch.sum(weights * self.activations, dim=1)  # Weight activations by importance

        # Apply ReLU to focus on features that have a positive influence
        heatmap = torch.maximum(heatmap, torch.tensor(0.0, device=heatmap.device))

        # Normalize the heatmap for visualization
        for t in range(heatmap.shape[0]):
            frame_heatmap = heatmap[t]
            max_val = torch.max(frame_heatmap) + 1e-8
            min_val = torch.min(frame_heatmap)
            if max_val > min_val:
                heatmap[t] = (frame_heatmap - min_val) / (max_val - min_val)
            else:
                heatmap[t] = torch.zeros_like(frame_heatmap)

        return heatmap.cpu().numpy()

    def __del__(self):
        # Remove hooks when the object is deleted
        for handle in self.hook_handles:
            handle.remove()

def extract_gradcam_heatmaps(model, target_layer_slow, target_layer_fast, inputs):
    """
    Extract Grad-CAM heatmaps from specified layers of the SlowFast model.
    Args:
        model: SlowFast model.
        target_layer_slow: Target layer in the slow pathway.
        target_layer_fast: Target layer in the fast pathway.
        inputs: Input tensors [slow_tensor, fast_tensor].
    Returns:
        dict: Grad-CAM heatmaps for slow and fast pathways.
    """
    # Create Grad-CAM objects
    gradcam_slow = GradCAM(model, target_layer_slow)
    gradcam_fast = GradCAM(model, target_layer_fast)
    
    # Clone inputs to ensure we don't modify the original tensors
    inputs_clone = [inp.clone() for inp in inputs]
    
    try:
        # Compute Grad-CAM heatmaps
        heatmap_slow = gradcam_slow(inputs_clone)  # Shape: (T_slow, H, W)
        heatmap_fast = gradcam_fast(inputs_clone)  # Shape: (T_fast, H, W)
        
        return {'slow': heatmap_slow, 'fast': heatmap_fast}
    except Exception as e:
        logger.error(f"Error in extract_gradcam_heatmaps: {e}")
        # Return empty heatmaps on error
        return {
            'slow': np.zeros((8, 7, 7)),  # Typical size for slow pathway
            'fast': np.zeros((32, 7, 7))  # Typical size for fast pathway
        }

# Function to visualize Grad-CAM heatmaps
def visualize_gradcam_heatmaps(heatmaps, slow_frames, fast_frames, video_id, output_dir):
    """
    Visualize Grad-CAM heatmaps alongside cropped frames.
    Args:
        heatmaps (dict): Grad-CAM heatmaps for slow and fast pathways.
        slow_frames (list): Original slow pathway frames.
        fast_frames (list): Original fast pathway frames.
        video_id (str): Video identifier.
        output_dir (str): Directory to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Visualize slow pathway
    for t in range(len(slow_frames)):
        if t >= heatmaps['slow'].shape[0]:
            continue  # Skip if frame index exceeds heatmap dimensions
            
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        frame = np.squeeze(slow_frames[t])
        plt.imshow(frame)
        plt.title(f"Slow Cropped Frame {t+1}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        heatmap = heatmaps['slow'][t]  # Shape: (H, W)
        # Safely resize heatmap to match frame dimensions
        try:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)  # Smooth to reduce noise
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            plt.imshow(frame, alpha=0.5)  # Overlay the heatmap on the original frame
        except Exception as e:
            logger.warning(f"Error visualizing slow heatmap {t}: {e}")
            plt.imshow(frame)  # Just show the frame if visualization fails
            
        plt.title(f"Slow Grad-CAM {t+1}")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_slow_gradcam_{t+1}.png"))
        plt.close()

    # Visualize fast pathway
    for t in range(len(fast_frames)):
        if t >= heatmaps['fast'].shape[0]:
            continue  # Skip if frame index exceeds heatmap dimensions
            
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        frame = np.squeeze(fast_frames[t])
        plt.imshow(frame)
        plt.title(f"Fast Cropped Frame {t+1}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        heatmap = heatmaps['fast'][t]  # Shape: (H, W)
        # Safely resize heatmap to match frame dimensions
        try:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)  # Smooth to reduce noise
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            plt.imshow(frame, alpha=0.5)  # Overlay the heatmap on the original frame
        except Exception as e:
            logger.warning(f"Error visualizing fast heatmap {t}: {e}")
            plt.imshow(frame)  # Just show the frame if visualization fails
            
        plt.title(f"Fast Grad-CAM {t+1}")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_fast_gradcam_{t+1}.png"))
        plt.close()

# Function to perform inference and visualize Grad-CAM heatmaps
@torch.no_grad()

def perform_inference(test_loader, model, cfg):
    """
    Perform inference on the test set and save features from different layers.
    Args:
        test_loader: DataLoader for the test set.
        model: SlowFast model.
        cfg: SlowFast configuration object.
    """
    # Define layers to extract features from
    layers_to_extract = {
        'low': {
            'slow': model.s2.pathway0_res0,
            'fast': model.s2.pathway1_res0
        },
        'mid': {
            'slow': model.s3.pathway0_res2,
            'fast': model.s3.pathway1_res2
        },
        'deep': {
            'slow': model.s5.pathway0_res0,
            'fast': model.s5.pathway1_res0
        }
    }

    # Create directories for feature outputs
    if SAVE_HEATMAPS_FEATURES:
        for depth in layers_to_extract:
            os.makedirs(os.path.join(cfg.OUTPUT_DIR, f"features_{depth}"), exist_ok=True)
    
    # Create a directory for final features
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "features"), exist_ok=True)
    
    # Create hooks to capture intermediate layer features
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            # For SlowFast pathways, handle different output formats
            if isinstance(output, list) or isinstance(output, tuple):
                # Store each pathway separately
                activation[f"{name}_pathway0"] = output[0].detach().cpu()
                activation[f"{name}_pathway1"] = output[1].detach().cpu()
            else:
                activation[name] = output.detach().cpu()
        return hook
    
    # Register hooks for each layer if we're saving features
    hook_handles = []
    if SAVE_HEATMAPS_FEATURES:
        for depth, layers in layers_to_extract.items():
            hook_handles.append(layers['slow'].register_forward_hook(get_activation(f"{depth}_slow")))
            hook_handles.append(layers['fast'].register_forward_hook(get_activation(f"{depth}_fast")))
    
    # Track class predictions and their confidences
    predictions = []
    device = next(model.parameters()).device
    
    # Add tracking counters
    total_samples = len(test_loader.dataset)
    processed_samples = 0
    
    for inputs, video_ids, slow_frames, fast_frames, labels in test_loader:
        batch_size = labels.size(0)  # Get current batch size
        processed_samples += batch_size
        
        logger.info(f"Processing batch {processed_samples-batch_size+1}-{processed_samples}/{total_samples}")
        
        # Process inputs
        try:
            inputs = [inp.to(device, non_blocking=True) for inp in inputs]
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass for predictions and feature extraction
            model.eval()
            with torch.no_grad():  # No need for gradients during inference
                preds, feat = model(inputs)
            
            # Process each sample in the batch
            for i in range(batch_size):
                sample_id = video_ids[i]
                pred_class = torch.argmax(preds[i]).item()
                confidence = torch.softmax(preds[i:i+1], dim=1)[0, pred_class].item()
                true_class = labels[i].item()
                
                # Store prediction
                predictions.append({
                    'video_id': sample_id,
                    'predicted': pred_class,
                    'true': true_class,
                    'confidence': confidence,
                    'correct': pred_class == true_class
                })
                
                # Save features if enabled
                if SAVE_HEATMAPS_FEATURES:
                    # Save final features
                    sample_feat = feat[i:i+1].cpu().numpy()
                    np.save(os.path.join(cfg.OUTPUT_DIR, "features", f"{sample_id}_final_features.npy"), sample_feat)
                    
                    # Save intermediate features from each layer
                    for depth in layers_to_extract:
                        features_dir = os.path.join(cfg.OUTPUT_DIR, f"features_{depth}")
                        
                        try:
                            # Check if the activations were captured
                            if f"{depth}_slow" in activation and f"{depth}_fast" in activation:
                                # Extract features for this sample
                                slow_feat = activation[f"{depth}_slow"][i].cpu().numpy()
                                fast_feat = activation[f"{depth}_fast"][i].cpu().numpy()
                                
                                # Create feature dictionary
                                feat_dict = {
                                    'slow': slow_feat,
                                    'fast': fast_feat,
                                    'video_id': sample_id,
                                    'label': true_class,
                                    'prediction': pred_class,
                                    'confidence': confidence,
                                }
                                
                                # Save as pickle file
                                with open(os.path.join(features_dir, f"{sample_id}_{depth}_features.pkl"), 'wb') as f:
                                    pickle.dump(feat_dict, f)
                                
                                logger.info(f"Saved {depth} features for {sample_id}")
                        except Exception as e:
                            logger.error(f"Error saving {depth} features for {sample_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Memory cleanup
        del inputs, labels
        if 'preds' in locals(): del preds
        if 'feat' in locals(): del feat
        # Clear activations after processing the batch
        activation.clear()
        torch.cuda.empty_cache()
    
    # Clean up hooks if they were registered
    for handle in hook_handles:
        handle.remove()
    
    # Save results
    try:
        import pandas as pd
        pd.DataFrame(predictions).to_csv(os.path.join(cfg.OUTPUT_DIR, "test_predictions.csv"), index=False)
        
        # Calculate accuracy
        if predictions:
            accuracy = sum(p['correct'] for p in predictions) / len(predictions)
            logger.info(f"Test accuracy: {accuracy:.4f} ({len(predictions)}/{total_samples} samples processed)")
            
            # Log detailed metrics by class
            class_metrics = {}
            for p in predictions:
                true_class = p['true']
                if true_class not in class_metrics:
                    class_metrics[true_class] = {'total': 0, 'correct': 0}
                class_metrics[true_class]['total'] += 1
                class_metrics[true_class]['correct'] += 1 if p['correct'] else 0
            
            # Print per-class accuracy
            for cls, metrics in class_metrics.items():
                cls_acc = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
                logger.info(f"Class {cls} accuracy: {cls_acc:.4f} ({metrics['correct']}/{metrics['total']})")
                
    except Exception as e:
        logger.error(f"Error saving prediction summary: {e}")


def train(cfg, train_loader, val_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = modify_slowfast_head(model, num_classes=2, device=device)

    # Freeze s1, s2, s3 layers
    for name, param in model.named_parameters():
        if any(s in name for s in ["s1", "s2", "s3"]):
            param.requires_grad = False

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer setup
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=1e-4, weight_decay=1e-5)
    
    # Loss function - with or without class weighting
    if CLASS_IMBALANCE:
        # Calculate class weights (adjust based on your actual class distribution)
        class_counts = [633, 1135]  # Your class counts from the results
        class_weights = torch.tensor([1/c for c in class_counts], device=device)
        class_weights = class_weights / class_weights.sum() * 2  # Normalize
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using weighted loss with class weights: {class_weights}")
    else:
        # Standard loss
        criterion = torch.nn.CrossEntropyLoss()
        logger.info("Using standard loss without class weighting")
    
    # Scheduler setup
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                         factor=0.5, patience=2, verbose=True)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training configuration
    num_epochs = 10
    best_val_acc = 0.0
    patience = 5  # Early stopping patience
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Gradient accumulation setup for effective larger batch size
    target_batch_size = 8  # Target effective batch size
    actual_batch_size = cfg.TRAIN.BATCH_SIZE if hasattr(cfg.TRAIN, 'BATCH_SIZE') else 4
    
    # Ensure we don't divide by zero
    if actual_batch_size <= 0:
        logger.warning(f"Invalid batch size: {actual_batch_size}. Setting to 1.")
        actual_batch_size = 1
        
    accumulation_steps = max(1, target_batch_size // actual_batch_size)
    logger.info(f"Using gradient accumulation: {accumulation_steps} steps (batch size {actual_batch_size})")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = train_correct = train_total = 0
        optimizer.zero_grad()  # Zero gradients before epoch starts
        
        for i, (inputs, video_id, _, _, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs = [inp.to(device) for inp in inputs]
            labels = labels.to(device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels) / accumulation_steps  # Scale for accumulation
                
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights after accumulation_steps or at the end of epoch
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Track metrics (scale loss back for reporting)
            train_loss += loss.item() * accumulation_steps
            _, pred = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()
            
            # Clean up memory
            del inputs, outputs, labels, loss
            torch.cuda.empty_cache()

        train_acc = 100 * train_correct / train_total
        logger.info(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Acc {train_acc:.2f}%")
        
        # Validation phase
        model.eval()
        val_loss = val_correct = val_total = 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for inputs, video_id, _, _, labels in val_loader:
                inputs = [inp.to(device) for inp in inputs]
                labels = labels.to(device)
                
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Save predictions for analysis
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
                # Clean up memory
                del inputs, outputs, labels
                torch.cuda.empty_cache()
        
        val_acc = 100 * val_correct / val_total
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Update scheduler based on validation accuracy
        scheduler.step(val_acc)
        
        # Track history
        history['train_loss'].append(train_loss/len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss/len(val_loader))
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'scaler': scaler.state_dict()
            }, os.path.join(cfg.OUTPUT_DIR, "slowfast_finetuned_best.pt"))
            logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Save checkpoint
        if (epoch + 1) % 2 == 0:  # Save every 2 epochs
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'scaler': scaler.state_dict(),
                'history': history
            }, os.path.join(cfg.OUTPUT_DIR, f"slowfast_checkpoint_epoch{epoch+1}.pt"))
    
    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'history': history
    }, os.path.join(cfg.OUTPUT_DIR, "slowfast_finetuned_final.pt"))
    
    # Plot and save training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "training_history.png"))
    plt.close()
    
    # Load best model for return
    best_checkpoint = torch.load(os.path.join(cfg.OUTPUT_DIR, "slowfast_finetuned_balanced_weight_top_86.pt"))
    model.load_state_dict(best_checkpoint['state_dict'])
    logger.info(f"Loaded best model with validation accuracy: {best_checkpoint['best_val_acc']:.2f}%")
    
    return model



def create_balanced_sampler(train_segments):
    """
    Create a sampler that balances classes in each batch.
    Args:
        train_segments: List of segments with their labels
    Returns:
        A balanced sampler
    """
    # Extract labels for each sample
    labels = [seg['label'] for seg in train_segments]
    
    # Count samples per class
    class_count = {}
    for label in labels:
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    
    # Calculate weights for each sample
    weights = []
    for label in labels:
        class_weight = 1.0 / class_count[label]
        weights.append(class_weight)
    
    # Create weighted sampler
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    logger.info(f"Created balanced sampler with class distribution: {class_count}")
    return sampler

def test(cfg):
    """
    Main function to fine-tune the SlowFast model and perform inference.
    Args:
        cfg: SlowFast configuration object.
    """
    # Set random seeds for reproducibility
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Test with config:")
    logger.info(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Build and initialize the model
    model = build_model(cfg)
    model = model.to(device)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    
    # Check if we're using a pre-trained fine-tuned model or doing fine-tuning
    if USE_PRETRAINED_FINETUNED:
        # Path to the fine-tuned model
        finetuned_model_path = os.path.join(cfg.OUTPUT_DIR, "slowfast_finetuned_balanced_weight_top_86.pt")
        
        if os.path.exists(finetuned_model_path):
            logger.info(f"Loading fine-tuned model from {finetuned_model_path}")
            # First load the pre-trained weights
            cu.load_test_checkpoint(cfg, model)
            # Then prepare model head for 2 classes
            model = modify_slowfast_head(model, num_classes=2, device=device)
            
            # Load the fine-tuned weights with error handling
            try:
                # Load checkpoint which contains state_dict
                checkpoint = torch.load(finetuned_model_path, map_location=device)
                # Extract state_dict from the checkpoint
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    logger.info("Successfully extracted state_dict from checkpoint")
                else:
                    state_dict = checkpoint  # Assume it's directly the state_dict
                
                # Try loading with strict=False to allow partial loading
                logger.info("Attempting to load fine-tuned weights")
                model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded fine-tuned weights")
                
                # Log validation accuracy if available
                if isinstance(checkpoint, dict) and 'best_val_acc' in checkpoint:
                    logger.info(f"Loaded model with validation accuracy: {checkpoint['best_val_acc']:.2f}%")
            except Exception as e:
                logger.error(f"Error loading fine-tuned weights: {e}")
                logger.info("Falling back to the pre-trained model with modified head")
        else:
            logger.warning(f"Fine-tuned model not found at {finetuned_model_path}. Loading default checkpoint.")
            cu.load_test_checkpoint(cfg, model)
            model = modify_slowfast_head(model, num_classes=2, device=device)
    else:
        # Load the pre-trained weights for fine-tuning
        logger.info("Loading pre-trained weights for fine-tuning")
        cu.load_test_checkpoint(cfg, model)

    # Define paths
    pickle_dir = "D:/pickle_dir/fine_tune"
    bbox_dir = "D:/frcnn_bboxes/bboxes_top"
    pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))

    logger.info(f"Found {len(pickle_files)} pickle files to process.")
    logger.info("----------------------------------------------------------")

    # Collect all_segments as before...
    all_segments = []
    for pkl_file in tqdm(pickle_files, desc="Collecting segments"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading pickle file {pkl_file}: {e}")
            continue

        for camera_id in data:
            if camera_id == 'cam3':
                for segments_group in data[camera_id]:
                    for segment in segments_group:
                        if 'segment_ratings' not in segment:
                            logger.warning(f"Skipping segment in {pkl_file} with missing 'segment_ratings'")
                            continue
                        rating = segment['segment_ratings'].get('t1', None)
                        try:
                            rating = int(rating)
                            if rating not in [2, 3]:
                                continue
                            label = 0 if rating == 2 else 1  # Map 2 -> 0, 3 -> 1
                        except (ValueError, TypeError):
                            logger.warning(f"Skipping segment in {pkl_file} with invalid rating: {rating}")
                            continue
                        video_id = (f"patient_{segment['patient_id']}_task_{segment['activity_id']}_"
                                    f"{segment['CameraId']}_seg_{segment['segment_id']}")
                        all_segments.append({
                            'frames': segment['frames'],
                            'video_id': video_id,
                            'label': label
                        })

    # If we're using a pre-trained model, use all data for inference
    # Otherwise, split for training
    if USE_PRETRAINED_FINETUNED:
        logger.info(f"Using all {len(all_segments)} segments for inference with pre-trained model")
        inference_segments = all_segments
        train_segments = []  # Empty, not used
    else:
        # Split into train/validation sets
        train_segments, inference_segments = train_test_split(
            all_segments, test_size=0.2, random_state=42
        )
        logger.info(f"Train/Val split: {len(train_segments)}/{len(inference_segments)} segments")

    # ─── Create Datasets ────────────────────────────────────────────────────────
    # Only create training datasets if we're going to fine-tune
    if not USE_PRETRAINED_FINETUNED:
        train_datasets = [
            SinglePickleFrameDataset(
                frames=seg["frames"],
                video_id=seg["video_id"],
                label=seg["label"],
                cfg=cfg,
                bbox_dir=bbox_dir,
                is_train=True
            )
            for seg in train_segments
        ]
    else:
        train_datasets = []  # Empty list, not used

    # Always create inference datasets
    inference_datasets = [
        SinglePickleFrameDataset(
            frames=seg["frames"],
            video_id=seg["video_id"],
            label=seg["label"],
            cfg=cfg,
            bbox_dir=bbox_dir,
            is_train=False
        )
        for seg in inference_segments
    ]

    # After creating all datasets, save the list of missing bounding box files
    if hasattr(SinglePickleFrameDataset, 'missing_bbox_files') and SinglePickleFrameDataset.missing_bbox_files:
        missing_files_path = os.path.join(cfg.OUTPUT_DIR, "missing_bbox_files.txt")
        with open(missing_files_path, 'w') as f:
            for file in set(SinglePickleFrameDataset.missing_bbox_files):  # Use set to remove duplicates
                f.write(f"{file}\n")
        logger.info(f"Saved list of {len(set(SinglePickleFrameDataset.missing_bbox_files))} missing bounding box files to {missing_files_path}")
    
    # ─── DataLoaders with custom collate_fn ─────────────────────────────────────
    # Only create training dataloader if we're fine-tuning
    if not USE_PRETRAINED_FINETUNED:
        if BALANCED_SAMPLING:
            # Create balanced sampler for training
            train_sampler = create_balanced_sampler(train_segments)
            
            # Use the sampler in the training DataLoader
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset(train_datasets),
                batch_size=4,
                sampler=train_sampler,  # Use the sampler instead of shuffle
                shuffle=False,  # Must be False when using a sampler
                num_workers=0,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=False,
                collate_fn=fastslow_collate_fn,
            )
            logger.info("Using balanced sampling for training batches")
        else:
            # Original DataLoader with shuffling
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset(train_datasets),
                batch_size=4,
                shuffle=True,
                num_workers=0,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=False,
                collate_fn=fastslow_collate_fn,
            )
            logger.info("Using standard random sampling for training batches")
    else:
        train_loader = None  # Not used

    # Always create inference dataloader
    inference_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(inference_datasets),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=False,
        collate_fn=fastslow_collate_fn,
    )

    # Fine‑tune & inference - only do fine-tuning if not using pre-trained model
    if not USE_PRETRAINED_FINETUNED:
        logger.info("Starting fine-tuning process...")
        model = train(cfg, train_loader, inference_loader, model)
        
    # Run inference
    logger.info("Starting inference process...")
    perform_inference(inference_loader, model, cfg)

def main():
    args = parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_files[0])
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()

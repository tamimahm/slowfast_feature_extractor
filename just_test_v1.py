#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Configuration flag: 
# 0 = do fine-tuning and then testing
# 1 = load saved fine-tuned model and run testing only
USE_PRETRAINED_FINETUNED = 0  # Change this value to control the workflow
# Add this at the top of your code with other global flags
CLASS_IMBALANCE = 1  # Set to 1 to enable class weight balancing, 0 for original approach
# Add this at the top of your code with other global flags
BALANCED_SAMPLING = 0  # Set to 1 to enable balanced batch sampling, 0 for original approach
# Add this at the top of your code with other global flags
SAVE_HEATMAPS_FEATURES = 1  # Set to 1 to save heatmaps and features, 0 to ignore
num_class=3  ##########set this to the correct number fo classes 0,1,2
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
import pandas as pd
from datetime import datetime
import torch.nn.functional as F
from skimage.transform import resize
from sklearn.metrics import precision_score, recall_score
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
# Path to the CSV file with camera assignments
ipsi_contra_csv = "D:\\nature_everything\\camera_assignments.csv"
camera_box="bboxes_ipsi"
_map = {1: 0, 2: 1, 3: 2}##########change this for the correct mapping
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
    def __init__(self, frames, video_id, label, camera_id, cfg, bbox_dir, is_train=True):
        self.frames = frames
        self.video_id = video_id
        self.label = label
        self.hand_id=camera_id
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
            if camera_box=='bboxes_top':
                y_max_ext = min(h, y_max + 30)
                x_min, x_max = max(0, x_min - 10), min(w, x_max + 10)
                y_min = max(0, y_min )
            else:
                if self.hand_id=='cam4':
                    y_max_ext = min(h, y_max - 20)
                    x_min, x_max = max(0, x_min - 30), min(w, x_max )
                    y_min = max(0, y_min ) 
                else:
                    y_max_ext = min(h, y_max - 20)
                    x_min, x_max = max(0, x_min ), min(w, x_max +30)
                    y_min = max(0, y_min )                                    
                    # # Create a figure to display both images
                    # plt.figure(figsize=(12, 6))
                    
                    # # Display original frame with bounding box
                    # plt.subplot(1, 2, 1)
                    # plt.imshow(frame)
                    # plt.title("Original Frame with Bounding Box")
                    # plt.axis('off')
                    
                    # # Display cropped frame
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(frame[y_min:y_max_ext, x_min:x_max])
                    # plt.title("Cropped Frame")
                    # plt.axis('off')
                    #plt.show()
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


_EPS = 1e-12

def _safe_entropy_from_probs(p: np.ndarray) -> float:
    p = np.clip(p.astype(np.float64), _EPS, 1.0)
    return float(-(p * np.log(p)).sum())

def parse_video_id_umera(video_id: str) -> dict:
    """
    Expected format from your code:
      patient_{pid}_task_{activity_id}_{CameraId}_seg_{segment_id}
    Example:
      patient_12_task_7_cam3_seg_4

    Returns a dictionary UMERA/BC can use for joins across views and granularities.
    """
    parts = video_id.split("_")
    out = {
        "video_id": video_id,
        "patient_id": None,
        "activity_id": None,
        "camera_id": None,
        "segment_id": None,
        "task_key": None,        # groups all views+segments for the same task trial
        "task_view_key": None,   # groups segments within the same view
    }
    try:
        # patient_{pid}_task_{activity}_camX_seg_{seg}
        pid = parts[1]
        activity = parts[3]
        cam = parts[4]
        seg = int(parts[-1]) if (len(parts) >= 7 and parts[-2] == "seg") else None

        out["patient_id"] = int(pid) if pid.isdigit() else pid
        out["activity_id"] = int(activity) if str(activity).isdigit() else activity
        out["camera_id"] = cam
        out["segment_id"] = seg

        # key to join ALL views for the same task (patient, task item)
        out["task_key"] = f"patient_{pid}_task_{activity}"

        # key to join segments within ONE view
        out["task_view_key"] = f"patient_{pid}_task_{activity}_{cam}"
    except Exception:
        # keep Nones; caller can still store raw video_id
        pass
    return out

def aggregate_task_level(records: list, num_classes: int) -> dict:
    """
    Build dataset-level task dictionaries so the Bayesian controller can load quickly:
      task_key -> { view -> {segment_id -> record}, task_prob_mean, task_entropy_mean, ... }
    """
    tasks = {}
    for r in records:
        tk = r.get("task_key")
        if tk is None:
            continue
        tasks.setdefault(tk, {"views": {}, "task_prob_mean": None, "task_entropy_mean": None})
        view = r.get("view", r.get("camera_id", "unknown"))
        tasks[tk]["views"].setdefault(view, {})
        seg_id = r.get("segment_id")
        tasks[tk]["views"][view][seg_id] = r

    # compute simple task-level aggregates (mean over all segment probs across all views)
    for tk, d in tasks.items():
        probs_list = []
        ent_list = []
        for view, seg_dict in d["views"].items():
            for seg_id, r in seg_dict.items():
                p = r.get("probs", None)
                if p is not None and len(p) == num_classes:
                    probs_list.append(p)
                    ent_list.append(r.get("entropy", _safe_entropy_from_probs(p)))
        if probs_list:
            P = np.stack(probs_list, axis=0)  # [n_clips, K]
            p_mean = P.mean(axis=0)
            d["task_prob_mean"] = p_mean
            d["task_entropy_mean"] = _safe_entropy_from_probs(p_mean)
        else:
            d["task_prob_mean"] = np.ones(num_classes, dtype=np.float32) / float(num_classes)
            d["task_entropy_mean"] = _safe_entropy_from_probs(d["task_prob_mean"])
    return tasks

# Function to perform inference and visualize Grad-CAM heatmaps
@torch.no_grad()

def perform_inference(test_loader, model, cfg, inference_segments=None,export_for_controller=True,controller_export_path=None,export_float16=True):
    """
    Perform inference on the test set and save features from different layers.
    Args:
        test_loader: DataLoader for the test set.
        model: SlowFast model.
        cfg: SlowFast configuration object.
        inference_segments: List of segments with both therapist ratings
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
            os.makedirs(os.path.join(cfg.OUTPUT_DIR, f"features_{camera_box.split('_')[1]}", f"features_{depth}"), exist_ok=True)
    
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
    
    # Create mapping from video_id to inference segment for quick lookup
    inference_map = {}
    if inference_segments:
        inference_map = {seg['video_id']: seg for seg in inference_segments}
    
    # Track class predictions and their confidences
    predictions = []
    controller_records = []
    num_classes = None  # inferred on first batch
    device = next(model.parameters()).device
    
    # Add tracking counters
    total_samples = len(test_loader.dataset)
    processed_samples = 0
    
    # Setup for task buffering across batches
    # Expected segment IDs
    SEG_SET = {0, 1, 2, 3}
    
    # Initialize task buffers for each depth
    task_buffers = {depth: {} for depth in layers_to_extract}
    
    # Track how many batches a task has been waiting
    batch_counts = {depth: {} for depth in layers_to_extract}
    
    # Configuration for segment processing
    min_segments_required = 4  # Minimum segments needed to process a task
    max_wait_batches = 10      # Maximum batches to wait before flushing incomplete tasks
    
    # Function to flush task segments to disk with improved robustness
    def flush_task(depth, task_key, save_dir, force=False):
        """
        Concatenate available segments and save to disk.
        
        Args:
            depth: Feature depth (low, mid, deep)
            task_key: Task identifier (patient_X_task_Y_camZ)
            save_dir: Directory to save features to
            force: Whether to force flushing even if not all segments are available
            
        Returns:
            Boolean indicating if the task was flushed
        """
        if task_key not in task_buffers[depth]:
            return False
            
        seg_dict = task_buffers[depth][task_key]
        available_segments = set(seg_dict.keys())
        
        # Check if all expected segments are available or if we should force flush
        if SEG_SET.issubset(available_segments) or (force and len(available_segments) >= min_segments_required):
            # Order the available segments
            available_segs = sorted(available_segments)
            seg_order = [seg_dict[k] for k in available_segs]
            
            # Concatenate available segments
            task_feat = np.concatenate(seg_order, axis=1)
            
            # Save the concatenated feature
            out_path = os.path.join(save_dir, f"{task_key}_taskfeat.npy")
            np.save(out_path, task_feat.astype(np.float32))
            
            # Log which segments were included
            seg_str = ','.join([str(s) for s in available_segs])
            logger.info(f"[{depth}] saved {os.path.basename(out_path)} with segments {seg_str}")
            
            # Remove the task from buffers
            task_buffers[depth].pop(task_key)
            if task_key in batch_counts[depth]:
                batch_counts[depth].pop(task_key)
                
            return True
        
        return False
    
    # Process each batch
    batch_counter = 0
    for inputs, video_ids, slow_frames, fast_frames, labels in test_loader:
        batch_size = labels.size(0)  # Get current batch size
        processed_samples += batch_size
        batch_counter += 1
        
        logger.info(f"Processing batch {batch_counter} ({processed_samples-batch_size+1}-{processed_samples}/{total_samples})")
        
        # Process inputs
        try:
            inputs = [inp.to(device, non_blocking=True) for inp in inputs]
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass for predictions and feature extraction
            model.eval()
            with torch.no_grad():  # No need for gradients during inference
                preds, feat = model(inputs)
                # infer num_classes once
                if num_classes is None:
                    # preds shape: [B, K]
                    num_classes = int(preds.shape[-1])

            
            # Process each sample in the batch
            for i in range(batch_size):
                sample_id = video_ids[i]
                pred_class = torch.argmax(preds[i]).item()
                confidence = torch.softmax(preds[i:i+1], dim=1)[0, pred_class].item()
                # --- extract controller-ready outputs ---
                logits_i = preds[i].detach().float().cpu().numpy()                # [K]
                probs_i = F.softmax(preds[i], dim=0).detach().float().cpu().numpy()  # [K]
                entropy_i = _safe_entropy_from_probs(probs_i)
                margin_i = float(np.sort(probs_i)[-1] - np.sort(probs_i)[-2]) if probs_i.size >= 2 else float(probs_i.max())
                # (your model returns feat; make it 1D numpy)
                feat_i = None
                try:
                    if isinstance(feat, (list, tuple)):
                        # if your model returns multiple feature tensors, store all
                        feat_i = [f[i].detach().float().cpu().numpy().ravel() for f in feat]
                    else:
                        feat_i = feat[i].detach().float().cpu().numpy().ravel()
                except Exception:
                    feat_i = None

                if export_float16:
                    logits_i = logits_i.astype(np.float16)
                    probs_i = probs_i.astype(np.float16)
                    if isinstance(feat_i, list):
                        feat_i = [x.astype(np.float16) for x in feat_i]
                    elif feat_i is not None:
                        feat_i = feat_i.astype(np.float16)

                # parse ids for joins (task_key joins all views; task_view_key joins segments within view)
                meta = parse_video_id_umera(sample_id)

                # attach available clinician labels (if present in inference_map)
                t1_label = None
                t2_label = None
                if sample_id in inference_map:
                    segment = inference_map[sample_id]
                    t1_label = segment.get("t1_label", None)
                    t2_label = segment.get("t2_label", None)

                # decide view tag (use camera_box or camera_id; camera_box is your run-time selector)
                # e.g., camera_box == "bboxes_top" -> "top"
                view_tag = camera_box.split("_")[-1] if isinstance(camera_box, str) else meta.get("camera_id", "unknown")

                controller_records.append({
                    # identifiers
                    "video_id": sample_id,
                    "patient_id": meta.get("patient_id"),
                    "activity_id": meta.get("activity_id"),
                    "camera_id": meta.get("camera_id"),
                    "segment_id": meta.get("segment_id"),
                    "task_key": meta.get("task_key"),
                    "task_view_key": meta.get("task_view_key"),
                    "view": view_tag,

                    # expert identity (this export is for SlowFast)
                    "expert": "SlowFast-R50",
                    "granularity": "segment",

                    # predictions
                    "pred": int(pred_class),
                    "confidence": float(confidence),
                    "entropy": float(entropy_i),
                    "margin": float(margin_i),
                    "logits": logits_i,
                    "probs": probs_i,

                    # features (for PoE fusion / reliability probes / downstream HBM)
                    "feat_global": feat_i,

                    # labels if available
                    "t1_label": None if t1_label is None else int(t1_label),
                    "t2_label": None if t2_label is None else int(t2_label),
                })


                # Get therapist ratings if available in inference_map
                if sample_id in inference_map:
                    segment = inference_map[sample_id]
                    t1_label = segment.get('t1_label')
                    t2_label = segment.get('t2_label')
                    
                    # Check if prediction matches either therapist rating
                    matches_t1 = (t1_label is not None) and (pred_class == t1_label)
                    matches_t2 = (t2_label is not None) and (pred_class == t2_label)
                    
                    # Consider correct if matches either rating
                    is_correct = matches_t1 or matches_t2
                    
                    # Store prediction with therapist rating info
                    predictions.append({
                        'video_id': sample_id,
                        'predicted': pred_class,
                        't1_label': t1_label,
                        't2_label': t2_label,
                        'confidence': confidence,
                        'correct': is_correct,
                        'matches_t1': matches_t1,
                        'matches_t2': matches_t2,
                        'matches_both': matches_t1 and matches_t2
                    })
                else:
                    # Fall back to dataset label if no therapist ratings available
                    true_class = labels[i].item()
                    predictions.append({
                        'video_id': sample_id,
                        'predicted': pred_class,
                        'true': true_class,
                        'confidence': confidence,
                        'correct': pred_class == true_class
                    })
                
                # Save features if enabled
                if SAVE_HEATMAPS_FEATURES:
                    # Save intermediate features from each layer
                    for depth in layers_to_extract:
                        features_dir = os.path.join(cfg.OUTPUT_DIR, f"features_{camera_box.split('_')[1]}", f"features_{depth}")
                        
                        try:
                            # Check if the activations were captured
                            if f"{depth}_slow" in activation and f"{depth}_fast" in activation:
                                # Extract features for this sample
                                slow_feat = activation[f"{depth}_slow"][i].cpu().numpy()
                                fast_feat = activation[f"{depth}_fast"][i].cpu().numpy()
                                
                                # -----------------------------------------------------------
                                # 1) spatial global‑average‑pool  (C, T, H, W) → (C, T)
                                slow_2d = slow_feat.mean(axis=(2, 3))          # (256, 8)
                                fast_2d = fast_feat.mean(axis=(2, 3))          # ( 32, 32)

                                # -----------------------------------------------------------
                                # 2) resample each pathway along time so BOTH => T = 20
                                def resample_to_20(arr):
                                    C, T = arr.shape
                                    # skimage.resize expects (T,) per channel so transpose twice
                                    return resize(arr.T, (20, C), mode="reflect", order=1,
                                                anti_aliasing=False, preserve_range=True).T  # (C,20)

                                slow_20 = resample_to_20(slow_2d)          # (256, 20)
                                fast_20 = resample_to_20(fast_2d)          # ( 32, 20)

                                # -----------------------------------------------------------
                                # 3) concat channels and save
                                clip_feat = np.concatenate([slow_20, fast_20], axis=0)   # (288,20)    
                                
                                # Parse video ID to get task and segment info
                                try:
                                    video_id = video_ids[i]
                                    parts = video_id.split('_')
                                    # Ensure format is consistent: patient_X_task_Y_camZ_seg_N
                                    if len(parts) >= 6 and parts[-2] == "seg":
                                        task_key = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"  # patient_X_task_Y_camZ
                                        seg_id = int(parts[-1])  # Extract segment ID
                                        
                                        # Add segment to buffer for this task
                                        if task_key not in task_buffers[depth]:
                                            task_buffers[depth][task_key] = {}
                                            batch_counts[depth][task_key] = 0
                                        
                                        task_buffers[depth][task_key][seg_id] = clip_feat
                                        # Reset batch counter when a new segment arrives
                                        batch_counts[depth][task_key] = 0
                                        
                                        # Try to flush if all segments are available
                                        flush_task(depth, task_key, features_dir)
                                        
                                        logger.debug(f"Added segment {seg_id} for task {task_key} (depth: {depth})")
                                    else:
                                        logger.warning(f"Unexpected video ID format: {video_id}")
                                except Exception as e:
                                    logger.error(f"Error parsing video ID {video_id}: {e}")
                        except Exception as e:
                            logger.error(f"Error saving {depth} features for {sample_id}: {e}")
            
            # After processing the batch, increment the wait counter for each task
            # and check if any task should be force-flushed
            for depth in layers_to_extract:
                for task_key in list(task_buffers[depth].keys()):
                    if task_key in batch_counts[depth]:
                        batch_counts[depth][task_key] += 1
                        
                        # Force flush tasks that have waited too long
                        if batch_counts[depth][task_key] >= max_wait_batches:
                            features_dir = os.path.join(cfg.OUTPUT_DIR, 
                                                  f"features_{camera_box.split('_')[1]}", 
                                                  f"features_{depth}")
                            
                            if flush_task(depth, task_key, features_dir, force=True):
                                logger.info(f"Forced flush of {task_key} after {batch_counts[depth][task_key]} batches")
            
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
    
    # At the end, flush any remaining tasks (with whatever segments are available)
    logger.info("Processing complete. Flushing any remaining tasks...")
    for depth in layers_to_extract:
        for task_key in list(task_buffers[depth].keys()):
            features_dir = os.path.join(cfg.OUTPUT_DIR, 
                              f"features_{camera_box.split('_')[1]}", 
                              f"features_{depth}")
            
            if flush_task(depth, task_key, features_dir, force=True):
                logger.info(f"Final forced flush of task {task_key}")
    
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
            
            # Calculate therapist agreement statistics if available
            therapist_stats = {
                't1_available': 0,
                't2_available': 0,
                't1_matches': 0,
                't2_matches': 0,
                'both_match': 0,
                'either_match': 0
            }
            
            for p in predictions:
                if 't1_label' in p and p['t1_label'] is not None:
                    therapist_stats['t1_available'] += 1
                    if p['matches_t1']:
                        therapist_stats['t1_matches'] += 1
                        
                if 't2_label' in p and p['t2_label'] is not None:
                    therapist_stats['t2_available'] += 1
                    if p['matches_t2']:
                        therapist_stats['t2_matches'] += 1
                
                if 't1_label' in p and 't2_label' in p and p['t1_label'] is not None and p['t2_label'] is not None:
                    if p['matches_both']:
                        therapist_stats['both_match'] += 1
                    if p['matches_t1'] or p['matches_t2']:
                        therapist_stats['either_match'] += 1
            
            # Print therapist agreement statistics
            if therapist_stats['t1_available'] > 0:
                t1_acc = therapist_stats['t1_matches'] / therapist_stats['t1_available']
                logger.info(f"Agreement with T1: {therapist_stats['t1_matches']}/{therapist_stats['t1_available']} ({t1_acc:.4f})")
            
            if therapist_stats['t2_available'] > 0:
                t2_acc = therapist_stats['t2_matches'] / therapist_stats['t2_available']
                logger.info(f"Agreement with T2: {therapist_stats['t2_matches']}/{therapist_stats['t2_available']} ({t2_acc:.4f})")
            
            # Calculate per-class metrics
            class_metrics = {}
            for p in predictions:
                # Determine true class - prioritize t1_label if available
                if 't1_label' in p and p['t1_label'] is not None:
                    true_class = p['t1_label']
                elif 't2_label' in p and p['t2_label'] is not None:
                    true_class = p['t2_label']
                else:
                    true_class = p.get('true')
                
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
        
    # Add segment-based accuracy analysis
    try:
        # Extract segment IDs from video IDs
        for p in predictions:
            # Extract segment ID (assuming format ends with "_seg_X")
            video_parts = p['video_id'].split('_seg_')
            if len(video_parts) > 1:
                p['segment_id'] = int(video_parts[1])  # Convert to integer for sorting
            else:
                p['segment_id'] = -1  # Default if no segment ID found

        # Group predictions by segment ID
        segment_results = {}
        for p in predictions:
            seg_id = p['segment_id']
            if seg_id not in segment_results:
                segment_results[seg_id] = {
                    0: {'correct': 0, 'total': 0, 'accuracy': 0.0},
                    1: {'correct': 0, 'total': 0, 'accuracy': 0.0}
                }
            
            # Determine true class for this prediction
            if 't1_label' in p and p['t1_label'] is not None:
                true_class = p['t1_label']
            elif 't2_label' in p and p['t2_label'] is not None:
                true_class = p['t2_label']
            else:
                true_class = p.get('true')
            
            # Update counts for this class
            if true_class in segment_results[seg_id]:
                segment_results[seg_id][true_class]['total'] += 1
                if p['correct']:
                    segment_results[seg_id][true_class]['correct'] += 1
        
        # Calculate accuracy for each segment and class
        for seg_id, classes in segment_results.items():
            for class_id, stats in classes.items():
                if stats['total'] > 0:
                    stats['accuracy'] = stats['correct'] / stats['total'] * 100
        
        # Create a DataFrame for better visualization
        import pandas as pd
        
        # Prepare data for DataFrame
        data = []
        for seg_id, classes in sorted(segment_results.items()):
            row = {
                'Segment ID': seg_id,
                'Class 0 Correct': classes[0]['correct'],
                'Class 0 Total': classes[0]['total'],
                'Class 0 Accuracy (%)': f"{classes[0]['accuracy']:.1f}",
                'Class 1 Correct': classes[1]['correct'],
                'Class 1 Total': classes[1]['total'],
                'Class 1 Accuracy (%)': f"{classes[1]['accuracy']:.1f}"
            }
            data.append(row)
        
        # Create DataFrame
        segment_df = pd.DataFrame(data)
        
        # Save to CSV
        segment_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "segment_accuracy.csv"), index=False)
        logger.info(f"Saved segment-based accuracy report to {os.path.join(cfg.OUTPUT_DIR, 'segment_accuracy.csv')}")
        
        # Print summary of segments with problematic accuracy
        problem_segments = []
        for row in data:
            for class_id in [0, 1, 2]:
                if row[f'Class {class_id} Total'] > 0 and float(row[f'Class {class_id} Accuracy (%)']) < 50:
                    problem_segments.append((row['Segment ID'], class_id, row[f'Class {class_id} Accuracy (%)']))
        
        if problem_segments:
            logger.info("Segments with accuracy below 50%:")
            for seg_id, class_id, accuracy in problem_segments:
                logger.info(f"  Segment {seg_id}, Class {class_id}: {accuracy}%")
        else:
            logger.info("All segments have accuracy above 50% for both classes")
            
    except Exception as e:
        logger.error(f"Error generating segment-based accuracy report: {e}")
        import traceback
        logger.error(traceback.format_exc())
    # =========================
    if export_for_controller:
        # choose path
        if controller_export_path is None:
            export_dir = os.path.join(cfg.OUTPUT_DIR, "controller_exports")
            os.makedirs(export_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            controller_export_path = os.path.join(export_dir, f"slowfast_{view_tag}_{ts}.pkl")

        tasks_export = aggregate_task_level(controller_records, num_classes=num_classes or 1)

        export_blob = {
            "meta": {
                "exported_at": datetime.now().isoformat(),
                "expert": "SlowFast-R50",
                "view": view_tag,
                "num_classes": int(num_classes or 0),
                # IMPORTANT: your current training maps original ratings 1/2/3 -> 0/1/2.
                # Store it explicitly so the controller can map back later.
                "label_mapping_note": "current script maps clinician segment ratings {1,2,3} -> classes {0,1,2}.",
                "cfg_output_dir": str(cfg.OUTPUT_DIR),
            },
            "segment_records": controller_records,  # list[dict]
            "task_aggregates": tasks_export,         # dict keyed by task_key
        }

        with open(controller_export_path, "wb") as f:
            pickle.dump(export_blob, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"[UMERA/BC] Exported SlowFast predictions+features to: {controller_export_path}")
        
    return predictions



def train(cfg, train_loader, val_loader, model, fold, class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = modify_slowfast_head(model, num_classes=num_class, device=device)

    # Freeze s1, s2, s3 layers
    for name, param in model.named_parameters():
        if any(s in name for s in ["s1", "s2", "s3"]):
            param.requires_grad = False

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer setup
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=1e-4, weight_decay=1e-4)
    
    # --- UPDATED LOSS FUNCTION BLOCK ---
    if CLASS_IMBALANCE and class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using weighted loss with dynamic class weights: {class_weights.tolist()}")
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        logger.info("Using standard loss without class weighting")
    
    # Scheduler setup
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                         factor=0.5, patience=2, verbose=True)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training configuration
    num_epochs = 15
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    # Training history - UPDATED to include P/R
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'train_precision': [], 'train_recall': [], # Stores list of arrays per epoch
        'val_precision': [], 'val_recall': []      # Stores list of arrays per epoch
    }
    
    target_batch_size = 8
    actual_batch_size = cfg.TRAIN.BATCH_SIZE if hasattr(cfg.TRAIN, 'BATCH_SIZE') else 4
    if actual_batch_size <= 0:
        actual_batch_size = 1
        
    accumulation_steps = max(1, target_batch_size // actual_batch_size)
    logger.info(f"Using gradient accumulation: {accumulation_steps} steps")
    
    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        train_loss = train_correct = train_total = 0
        train_preds, train_targets = [], [] # New: Collect train preds for P/R calculation
        optimizer.zero_grad()
        
        for i, (inputs, video_id, _, _, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs = [inp.to(device) for inp in inputs]
            labels = labels.to(device)
            
            with autocast():
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels) / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Metrics
            train_loss += loss.item() * accumulation_steps
            _, pred = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()
            
            # Collect predictions for P/R
            train_preds.extend(pred.detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())
            
            del inputs, outputs, labels, loss
            torch.cuda.empty_cache()

        train_acc = 100 * train_correct / train_total
        
        # Calculate Train P/R
        # average=None returns metrics for each class in a sorted array (class 0, 1, 2...)
        train_prec = precision_score(train_targets, train_preds, average=None, zero_division=0)
        train_rec = recall_score(train_targets, train_preds, average=None, zero_division=0)
        
        logger.info(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Acc {train_acc:.2f}%")
        logger.info(f"   Train Prec: {np.round(train_prec, 3)} | Recall: {np.round(train_rec, 3)}")
        
        # --- Validation phase ---
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
                
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
                del inputs, outputs, labels
                torch.cuda.empty_cache()
        
        val_acc = 100 * val_correct / val_total
        
        # Calculate Val P/R
        val_prec = precision_score(val_targets, val_preds, average=None, zero_division=0)
        val_rec = recall_score(val_targets, val_preds, average=None, zero_division=0)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"   Val Prec:   {np.round(val_prec, 3)} | Recall: {np.round(val_rec, 3)}")
        
        scheduler.step(val_acc)
        
        # Update history
        history['train_loss'].append(train_loss/len(train_loader))
        history['train_acc'].append(train_acc)
        history['train_precision'].append(train_prec)
        history['train_recall'].append(train_rec)
        
        history['val_loss'].append(val_loss/len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        
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
            logger.info(f"New best model saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history
            }, os.path.join(cfg.OUTPUT_DIR, f"slowfast_checkpoint_epoch{epoch+1}.pt"))
    
    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'history': history
    }, os.path.join(cfg.OUTPUT_DIR, "slowfast_finetuned_final.pt"))
    
    # Plotting (Updated for Accuracy only, but History contains everything)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "training_history_"+str(fold)+".png"))
    plt.close()
    
    # Load best model for return
    if camera_box=='bboxes_top':
        best_checkpoint = torch.load(os.path.join(cfg.OUTPUT_DIR, "slowfast_finetuned_balanced_weight_top_86.pt"))
    else:
        best_checkpoint = torch.load(os.path.join(cfg.OUTPUT_DIR, "slowfast_finetuned_best.pt"))
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
# Modified code to check if ratings from t1 and t2 match
def get_valid_rating(segment):
    """
    Get a valid rating only if t1 and t2 agree, or use the one that's available.
    Returns the rating if valid, or None if invalid.
    """
    t1_rating = segment['segment_ratings'].get('t1', None)
    t2_rating = segment['segment_ratings'].get('t2', None)
    
    # Try to convert ratings to integers
    try:
        t1_rating = int(t1_rating) if t1_rating is not None else None
    except (ValueError, TypeError):
        t1_rating = None
        
    try:
        t2_rating = int(t2_rating) if t2_rating is not None else None
    except (ValueError, TypeError):
        t2_rating = None
    
    # Decision logic
    if t1_rating is not None and t2_rating is not None:
        # Both ratings available - check if they match
        if t1_rating == t2_rating:
            return t1_rating  # They match, use either one
        else:
            return 'no_match'  # They don't match, invalid rating
    elif t1_rating is not None:
        # Only t1 is available
        return t1_rating
    elif t2_rating is not None:
        # Only t2 is available
        return t2_rating
    else:
        # No valid ratings
        return None
def perform_inference_fold(val_loader, model, fold_num):
    """
    Simplified inference function for fold validation.
    """
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    total_samples = len(val_loader.dataset)
    processed = 0
    
    with torch.no_grad():
        for inputs, video_ids, _, _, labels in val_loader:
            batch_size = len(video_ids)
            processed += batch_size
            
            # Move to device
            inputs = [inp.to(device) for inp in inputs]
            labels = labels.to(device)
            
            # Get predictions
            outputs, _ = model(inputs)
            
            # Process each prediction
            for i in range(batch_size):
                pred = torch.argmax(outputs[i]).item()
                true = labels[i].item()
                predictions.append({
                    'video_id': video_ids[i],
                    'predicted': pred,
                    'true': true,
                    'correct': pred == true
                })
    
    # Calculate accuracy
    accuracy = sum(p['correct'] for p in predictions) / len(predictions)
    
    # Calculate per-class metrics
    class_metrics = {}
    for p in predictions:
        true_class = p['true']
        if true_class not in class_metrics:
            class_metrics[true_class] = {'total': 0, 'correct': 0}
        class_metrics[true_class]['total'] += 1
        class_metrics[true_class]['correct'] += 1 if p['correct'] else 0
    
    # Return results
    return {
        'accuracy': accuracy,
        'class_metrics': class_metrics,
        'predictions': predictions
    }    

import re

def get_patient_id_from_video_id(video_id: str):
    """
    Extract patient id from strings like:
      patient_12_task_7_cam3_seg_4
    Returns int if possible, else string id.
    """
    m = re.search(r"patient_([^_]+)", video_id)
    if not m:
        return None
    pid = m.group(1)
    return int(pid) if pid.isdigit() else pid

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
        if camera_box=='bboxes_top':
            finetuned_model_path = os.path.join(cfg.OUTPUT_DIR,'final_top_run', "best_fold_model.pt")
        else:
            finetuned_model_path = os.path.join(cfg.OUTPUT_DIR, 'final_ipsi_run', "best_fold_model.pt")        
        if os.path.exists(finetuned_model_path):
            logger.info(f"Loading fine-tuned model from {finetuned_model_path}")
            # First load the pre-trained weights
            cu.load_test_checkpoint(cfg, model)
            # Then prepare model head for 2 classes
            model = modify_slowfast_head(model, num_classes=num_class, device=device)
            
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
            model = modify_slowfast_head(model, num_classes=num_class, device=device)
    else:
        # Load the pre-trained weights for fine-tuning
        logger.info("Loading pre-trained weights for fine-tuning")
        cu.load_test_checkpoint(cfg, model)

    # Define paths
    # if USE_PRETRAINED_FINETUNED==1:
    #     pickle_dir = "D:/pickle_dir"
    # else:
    ###########################################################################################change directory########################################################
    pickle_dir = "D:/nature_everything/nature_dataset"
    bbox_dir = "D:/nature_everything/frcnn_boxes/"+camera_box
    pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))
    # Read the CSV file
    camera_df = pd.read_csv(ipsi_contra_csv)
    # Create a dictionary mapping patient_id to ipsilateral_camera_id
    patient_to_ipsilateral = dict(zip(camera_df['patient_id'], camera_df['ipsilateral_camera_id']))
    logger.info(f"Found {len(pickle_files)} pickle files to process.")
    logger.info("----------------------------------------------------------")

    # Collect all_segments as before...
    all_segments = []
    inference_segments=[]
    r1=0
    r2=0
    no_match=0
    r3=0
    for pkl_file in tqdm(pickle_files, desc="Collecting segments"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading pickle file {pkl_file}: {e}")
            continue

        for camera_id in data:
            for segments_group in data[camera_id]:
                for segment in segments_group:
                    # Extract patient_id and camera_id from segment
                    patient_id = segment['patient_id']
                    segment_camera_id = segment['CameraId']
                    # Check if this camera is the ipsilateral camera for the patient
                    if camera_id=='cam3' and camera_box=='bboxes_top':
                        ipsilateral_camera=camera_id
                    elif camera_id!='cam3' and camera_box=='bboxes_ipsi':
                        ipsilateral_camera = patient_to_ipsilateral.get(patient_id)
                    else:
                        continue
                    if ipsilateral_camera == segment_camera_id:
                        if 'segment_ratings' not in segment:
                            logger.warning(f"Skipping segment in {pkl_file} with missing 'segment_ratings'")
                            continue
                        rating = get_valid_rating(segment)
                        t1_rating = segment['segment_ratings'].get('t1', None)
                        t2_rating = segment['segment_ratings'].get('t2', None)
                        if rating is None or rating not in [1,2, 3,"no_match"]: ###########this needs to be changed to read 1,2,3
                            continue
                        else:
                            t1_label = None if t1_rating is None else _map.get(t1_rating)
                            t2_label = None if t2_rating is None else _map.get(t2_rating)
                            video_id = (f"patient_{segment['patient_id']}_task_{segment['activity_id']}_"
                                        f"{segment['CameraId']}_seg_{segment['segment_id']}")
                            inference_segments.append({
                                'frames': segment['frames'],
                                'video_id': video_id,
                                't1_label':t1_label,
                                't2_label':t2_label,
                                'hand_id': ipsilateral_camera
                                                    })                        

                        try:
                            if rating =="no_match":
                                no_match+=1
                                continue
                            rating = int(rating)
                            
                            if rating == 1:
                                label = 0
                                r1 += 1
                            elif rating == 2:
                                label = 1
                                r2 += 1
                            else:
                                label = 2
                                r3 += 1
                        except (ValueError, TypeError):
                            logger.warning(f"Skipping segment in {pkl_file} with invalid rating: {rating}")
                            continue
                        all_segments.append({
                            'frames': segment['frames'],
                            'video_id': video_id,
                            'label': label,
                            'hand_id': ipsilateral_camera
                        })

    # If we're using a pre-trained model, use all data for inference
    # Otherwise, split for training
    if USE_PRETRAINED_FINETUNED:
        logger.info(f"Using all {len(inference_segments)} segments for inference with pre-trained model")
        
        # Create dataset from inference_segments
        inference_datasets = [
            SinglePickleFrameDataset(
                frames=seg["frames"],
                video_id=seg["video_id"],
                label=seg.get("t1_label", 0),  # Use t1_label as primary label
                camera_id=seg['hand_id'],
                cfg=cfg,
                bbox_dir=bbox_dir,
                is_train=False
            )
            for seg in inference_segments
        ]
        
        # Create inference dataloader
        inference_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(inference_datasets),
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False,
            collate_fn=fastslow_collate_fn,
        )
        
        # Run inference using the pre-trained model
        logger.info("Starting inference process...")
        perform_inference(inference_loader, model, cfg, inference_segments)
    else:
        # # Implement 5-fold cross-validation using all_segments
        # from sklearn.model_selection import KFold
        #kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        k_folds = 5
        groups = []
        bad = 0
        for seg in all_segments:
            pid = get_patient_id_from_video_id(seg["video_id"])
            if pid is None:
                bad += 1
                pid = f"unknown_{bad}"  # keep split running, but you should inspect these
            groups.append(pid)

        groups = np.asarray(groups)      
        
        # Track best model across folds
        best_val_acc = 0.0
        best_model_state = None
        best_fold = -1
        fold_results = []
        
        # Create a mapping from video_id to segment
        segment_map = {seg['video_id']: seg for seg in all_segments}       
        from sklearn.model_selection import StratifiedGroupKFold
        # 'y_labels' ensures class stratification
        groups = [get_patient_id_from_video_id(s["video_id"]) for s in all_segments]
        y_labels = [s['label'] for s in all_segments]

        # 2. Use StratifiedGroupKFold instead of GroupKFold
        sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

        # 3. Pass 'y=y_labels' and 'groups=groups' to the split function
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(all_segments, y=y_labels, groups=groups)):
            logger.info(f"Starting fold {fold+1}/{k_folds}")

            train_segments = [all_segments[i] for i in train_idx]
            val_segments   = [all_segments[i] for i in val_idx]

            # sanity check: no patient overlap
            train_p = {get_patient_id_from_video_id(s["video_id"]) for s in train_segments}
            val_p   = {get_patient_id_from_video_id(s["video_id"]) for s in val_segments}
            overlap = train_p.intersection(val_p)
            logger.info(f"Fold {fold+1}: patient overlap = {len(overlap)}")
            
            # Sanity check: Check label distribution in this fold
            train_lbls = [s['label'] for s in train_segments]
            val_lbls = [s['label'] for s in val_segments]
            logger.info(f"Fold {fold+1} Train distribution: {np.bincount(train_lbls, minlength=3)}")
            logger.info(f"Fold {fold+1} Val distribution:   {np.bincount(val_lbls, minlength=3)}")
            # --- NEW: Calculate Class Weights Dynamically (Keep this logic) ---
            weights_tensor = None
            if CLASS_IMBALANCE:
                # 1. Extract all labels from the current training split
                train_labels = [seg['label'] for seg in train_segments]               
                # 2. Count frequencies
                class_counts = np.bincount(train_labels, minlength=3)
                total_samples = len(train_labels)
                num_classes = 3               
                logger.info(f"Fold {fold+1} Class Counts: {class_counts}")
                # 3. Compute 'balanced' weights: N / (num_classes * count_i)
                weights = total_samples / (num_classes * (class_counts + 1e-6))
                # 4. Convert to tensor
                weights_tensor = torch.tensor(weights, dtype=torch.float)           
            # Calculate split percentages
            train_pct = len(train_segments) / len(all_segments) * 100
            val_pct = len(val_segments) / len(all_segments) * 100
            logger.info(f"Fold {fold+1} split: {len(train_segments)} ({train_pct:.1f}%) train, "
                        f"{len(val_segments)} ({val_pct:.1f}%) validation")
            
            # Create datasets for this fold
            train_datasets = [
                SinglePickleFrameDataset(
                    frames=seg["frames"],
                    video_id=seg["video_id"],
                    label=seg["label"],
                    camera_id=seg['hand_id'],
                    cfg=cfg,
                    bbox_dir=bbox_dir,
                    is_train=True
                )
                for seg in train_segments
            ]
            
            val_datasets = [
                SinglePickleFrameDataset(
                    frames=seg["frames"],
                    video_id=seg["video_id"],
                    label=seg["label"],
                    camera_id=seg['hand_id'],
                    cfg=cfg,
                    bbox_dir=bbox_dir,
                    is_train=False
                )
                for seg in val_segments
            ]
            
            # Create dataloaders
            if BALANCED_SAMPLING:
                train_sampler = create_balanced_sampler(train_segments)
                train_loader = torch.utils.data.DataLoader(
                    torch.utils.data.ConcatDataset(train_datasets),
                    batch_size=4,
                    sampler=train_sampler,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                    drop_last=False,
                    collate_fn=fastslow_collate_fn,
                )
            else:
                train_loader = torch.utils.data.DataLoader(
                    torch.utils.data.ConcatDataset(train_datasets),
                    batch_size=4,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                    drop_last=False,
                    collate_fn=fastslow_collate_fn,
                )
            
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset(val_datasets),
                batch_size=4,
                shuffle=False,
                num_workers=0,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=False,
                collate_fn=fastslow_collate_fn,
            )
            
            # Reset the model for this fold
            fold_model = build_model(cfg)
            fold_model = fold_model.to(device)
            cu.load_test_checkpoint(cfg, fold_model)
            
            # Train the model for this fold
            logger.info(f"Training model for fold {fold+1}")
            #fold_model = train(cfg, train_loader, val_loader, fold_model)
            # UPDATE THIS CALL:
            fold_model = train(cfg, train_loader, val_loader, fold_model, fold,class_weights=weights_tensor)
            # Save fold model
            fold_dir = os.path.join(cfg.OUTPUT_DIR, f"fold_{fold+1}")
            os.makedirs(fold_dir, exist_ok=True)
            torch.save({
                'fold': fold+1,
                'state_dict': fold_model.state_dict(),
            }, os.path.join(fold_dir, "fold_model.pt"))
            
            # Evaluate on validation set
            logger.info(f"Evaluating model for fold {fold+1}")
            
            # Perform inference on validation set
            val_results = perform_inference_fold(val_loader, fold_model, fold+1)
            fold_acc = val_results['accuracy']
            
            # Store fold results
            fold_results.append({
                'fold': fold+1,
                'accuracy': fold_acc,
                'class_metrics': val_results['class_metrics']
            })
            
            logger.info(f"Fold {fold+1} validation accuracy: {fold_acc:.4f}")
            
            # Track best model
            if fold_acc > best_val_acc:
                best_val_acc = fold_acc
                best_model_state = fold_model.state_dict()
                best_fold = fold+1
                
                # Save as best model
                torch.save({
                    'fold': fold+1,
                    'state_dict': fold_model.state_dict(),
                    'accuracy': fold_acc
                }, os.path.join(cfg.OUTPUT_DIR, "best_fold_model.pt"))
        
        # After all folds, print summary
        logger.info("Cross-validation complete. Summary of fold results:")
        fold_accs = [fold["accuracy"] for fold in fold_results]
        mean_acc = sum(fold_accs) / len(fold_accs)
        std_acc = (sum((acc - mean_acc) ** 2 for acc in fold_accs) / len(fold_accs)) ** 0.5
        
        logger.info(f"Mean accuracy across folds: {mean_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"Best fold: {best_fold} with accuracy {best_val_acc:.4f}")
        
        # Final evaluation using best model on all inference_segments
        logger.info(f"Using best model from fold {best_fold} for final inference")
        

        # First load the pre-trained weights
        cu.load_test_checkpoint(cfg, model)
        # Then prepare model head for 2 classes
        model = modify_slowfast_head(model, num_classes=num_class, device=device)
        best_model_path=os.path.join(cfg.OUTPUT_DIR, "best_fold_model.pt")
        # Load the fine-tuned weights with error handling
        try:
            # Load checkpoint which contains state_dict
            checkpoint = torch.load(best_model_path, map_location=device)
            # Extract state_dict from the checkpoint
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                logger.info("Successfully extracted state_dict from checkpoint")
            else:
                state_dict = checkpoint  # Assume it's directly the state_dict
            
            # Try loading with strict=False to allow partial loading
            logger.info("Attempting to load best fold model weights")
            model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded fine-tuned weights")
            
            # Log validation accuracy if available
            if isinstance(checkpoint, dict) and 'best_val_acc' in checkpoint:
                logger.info(f"Loaded model with validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        except Exception as e:
            logger.error(f"Error loading fine-tuned weights: {e}")
            logger.info("Falling back to the pre-trained model with modified head")

        # Create dataset from inference_segments for final evaluation
        inference_datasets = [
            SinglePickleFrameDataset(
                frames=seg["frames"],
                video_id=seg["video_id"],
                label=seg.get("t1_label", 0),  # Use t1_label as primary label
                camera_id=seg['hand_id'],
                cfg=cfg,
                bbox_dir=bbox_dir,
                is_train=False
            )
            for seg in inference_segments
        ]
        
        # Create inference dataloader
        inference_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(inference_datasets),
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False,
            collate_fn=fastslow_collate_fn,
        )
        
        # Run final inference
        logger.info("Starting final inference with best model...")
        perform_inference(inference_loader, model, cfg, inference_segments)

def main():
    args = parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_files[0])
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

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

# Function to load bounding boxes
def load_bboxes(bbox_dir, video_id):
    """
    Load bounding box data for a video segment.
    Args:
        bbox_dir (str): Directory containing bounding box files.
        video_id (str): Video identifier (e.g., 'patient_1_task_1_cam3_seg_0').
    Returns:
        list: List of bounding boxes, each an array of shape (4,) [x_min, y_min, x_max, y_max].
              Returns None if file not found or data is invalid.
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

# Custom Dataset for a Single Video Segment with Cropped Frames
class SinglePickleFrameDataset(torch.utils.data.Dataset):
    def __init__(self, frames, video_id, label, cfg, bbox_dir):
        """
        Initialize dataset for a single video segment.
        Args:
            frames (list): List of video frames.
            video_id (str): Unique identifier for the video segment.
            label (int): Label for the segment (0 or 1).
            cfg: SlowFast configuration object.
            bbox_dir (str): Directory containing bounding box files.
        """
        self.frames = frames
        self.video_id = video_id
        self.label = label  # Label for training (0 or 1)
        self.cfg = cfg
        self.bbox_dir = bbox_dir
        self.bboxes = load_bboxes(bbox_dir, video_id)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        num_frames_fast = self.cfg.DATA.NUM_FRAMES  # e.g., 8
        alpha = self.cfg.SLOWFAST.ALPHA  # e.g., 4
        num_frames_slow = num_frames_fast // alpha  # e.g., 2
        logger.debug(f"Video {self.video_id} - num_frames_fast: {num_frames_fast}, num_frames_slow: {num_frames_slow}")

        # Sample indices for fast pathway
        if len(self.frames) < num_frames_fast:
            fast_indices = list(range(len(self.frames))) + [len(self.frames) - 1] * (num_frames_fast - len(self.frames))
        else:
            step = max(1, len(self.frames) // num_frames_fast)
            fast_indices = [i for i in range(0, len(self.frames), step)][:num_frames_fast]

        # Sample indices for slow pathway
        if len(self.frames) < num_frames_slow:
            slow_indices = list(range(len(self.frames))) + [len(self.frames) - 1] * (num_frames_slow - len(self.frames))
        else:
            step = max(1, len(self.frames) // num_frames_slow)
            slow_indices = [i for i in range(0, len(self.frames), step)][:num_frames_slow]

        def crop_frame(frame, bbox):
            """Crop frame using bounding box with 30-pixel extension on the lower side."""
            if bbox is None or len(bbox) != 4:
                logger.warning(f"No valid bbox for frame in {self.video_id}, using full frame")
                return frame
            x_min, y_min, x_max, y_max = map(int, bbox)
            h, w = frame.shape[:2]
            y_max_extended = y_max + 30  # Extend lower side by 30 pixels
            x_min, x_max = max(0, x_min), min(w, x_max)
            y_min, y_max_extended = max(0, y_min), min(h, y_max_extended)
            if x_max <= x_min or y_max_extended <= y_min:
                logger.warning(f"Invalid crop coordinates [{x_min}, {y_min}, {x_max}, {y_max_extended}] for {self.video_id}, using full frame")
                return frame
            return frame[y_min:y_max_extended, x_min:x_max]

        fast_processed = []
        fast_original = []
        slow_processed = []
        slow_original = []

        # Process frames for fast pathway
        for i in fast_indices:
            frame = self.frames[i]
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).numpy()
            elif isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame.squeeze(0)

            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

            if frame.ndim != 3 or frame.shape[2] != 3:
                logger.warning(f"Invalid frame dimensions {frame.shape} for {self.video_id}, using dummy")
                frame = np.zeros((224, 224, 3), dtype=np.uint8)

            bbox = self.bboxes[i] if self.bboxes and i < len(self.bboxes) else None
            cropped_frame = crop_frame(frame, bbox)
            pil_img = Image.fromarray(cropped_frame)
            fast_processed.append(self.transform(pil_img))
            fast_original.append(cropped_frame)

        # Process frames for slow pathway
        for i in slow_indices:
            frame = self.frames[i]
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).numpy()
            elif isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame.squeeze(0)

            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

            if frame.ndim != 3 or frame.shape[2] != 3:
                logger.warning(f"Invalid frame dimensions {frame.shape} for {self.video_id}, using dummy")
                frame = np.zeros((224, 224, 3), dtype=np.uint8)

            bbox = self.bboxes[i] if self.bboxes and i < len(self.bboxes) else None
            cropped_frame = crop_frame(frame, bbox)
            pil_img = Image.fromarray(cropped_frame)
            slow_processed.append(self.transform(pil_img))
            slow_original.append(cropped_frame)

        # Stack tensors
        if not fast_processed or not slow_processed:
            logger.warning(f"No valid frames for {self.video_id}, using dummy tensors")
            fast_tensor = torch.zeros((3, num_frames_fast, 224, 224))
            slow_tensor = torch.zeros((3, num_frames_slow, 224, 224))
            fast_original = [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames_fast
            slow_original = [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames_slow
        else:
            fast_tensor = torch.stack(fast_processed, dim=0).permute(1, 0, 2, 3)
            slow_tensor = torch.stack(slow_processed, dim=0).permute(1, 0, 2, 3)
            if fast_tensor.shape[1] < num_frames_fast:
                fast_tensor = torch.cat([fast_tensor, fast_tensor[:, -1:].repeat(1, num_frames_fast - fast_tensor.shape[1], 1, 1)], dim=1)
                fast_original.extend([fast_original[-1]] * (num_frames_fast - len(fast_original)))
            if slow_tensor.shape[1] < num_frames_slow:
                slow_tensor = torch.cat([slow_tensor, slow_tensor[:, -1:].repeat(1, num_frames_slow - slow_tensor.shape[1], 1, 1)], dim=1)
                slow_original.extend([slow_original[-1]] * (num_frames_slow - len(slow_original)))
            fast_tensor = fast_tensor[:, :num_frames_fast]
            slow_tensor = slow_tensor[:, :num_frames_slow]
            fast_original = fast_original[:num_frames_fast]
            slow_original = slow_original[:num_frames_slow]

        logger.debug(f"Video {self.video_id} - fast_tensor shape: {fast_tensor.shape}, slow_tensor shape: {slow_tensor.shape}")
        video_tensor = [slow_tensor, fast_tensor]
        return video_tensor, self.video_id, slow_original, fast_original, torch.tensor(self.label, dtype=torch.long)

# Function to modify the SlowFast head
def modify_slowfast_head(model, num_classes, device):
    """
    Modify the SlowFast model's head to match the number of classes.
    Args:
        model: SlowFast model.
        num_classes (int): Number of classes for the new task.
        device: Device to place the new layer on (e.g., 'cuda:0' or 'cpu').
    Returns:
        model: Modified SlowFast model.
    """
    in_features = model.head.projection.in_features  # e.g., 2304 for SlowFast R50
    model.head.projection = torch.nn.Linear(in_features, num_classes).to(device)
    logger.info(f"Modified SlowFast head to output {num_classes} classes on device {device}")
    return model

# Grad-CAM implementation for SlowFast
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
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, inputs, target_class=None):
        """
        Compute Grad-CAM heatmap for the target class.
        Args:
            inputs: Input tensors [slow_tensor, fast_tensor].
            target_class: Target class index for Grad-CAM. If None, use the predicted class.
        Returns:
            heatmap: Grad-CAM heatmap of shape (T, H, W).
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        outputs, _ = self.model(inputs)
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        # Backward pass to get gradients
        outputs[:, target_class].backward()

        # Compute Grad-CAM
        gradients = self.gradients  # Shape: (B, C, T, H, W)
        activations = self.activations  # Shape: (B, C, T, H, W)

        # Average gradients over the channel dimension
        weights = torch.mean(gradients, dim=[0, 3, 4], keepdim=True)  # Shape: (1, C, T, 1, 1)
        heatmap = torch.sum(weights * activations, dim=1).squeeze()  # Shape: (T, H, W)

        # Apply ReLU
        heatmap = torch.maximum(heatmap, torch.tensor(0.0, device=heatmap.device))

        # Normalize the heatmap
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

# Function to extract Grad-CAM heatmaps
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
    gradcam_slow = GradCAM(model, target_layer_slow)
    gradcam_fast = GradCAM(model, target_layer_fast)

    # Compute Grad-CAM heatmaps
    heatmap_slow = gradcam_slow(inputs)  # Shape: (T_slow, H, W)
    heatmap_fast = gradcam_fast(inputs)  # Shape: (T_fast, H, W)

    return {'slow': heatmap_slow, 'fast': heatmap_fast}

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
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        frame = np.squeeze(slow_frames[t])
        plt.imshow(frame)
        plt.title(f"Slow Cropped Frame {t+1}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        heatmap = heatmaps['slow'][t]  # Shape: (H, W)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)  # Smooth to reduce noise
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.imshow(frame, alpha=0.5)  # Overlay the heatmap on the original frame
        plt.title(f"Slow Grad-CAM {t+1}")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_slow_gradcam_{t+1}.png"))
        plt.close()

    # Visualize fast pathway
    for t in range(len(fast_frames)):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        frame = np.squeeze(fast_frames[t])
        plt.imshow(frame)
        plt.title(f"Fast Cropped Frame {t+1}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        heatmap = heatmaps['fast'][t]  # Shape: (H, W)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)  # Smooth to reduce noise
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.imshow(frame, alpha=0.5)  # Overlay the heatmap on the original frame
        plt.title(f"Fast Grad-CAM {t+1}")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_fast_gradcam_{t+1}.png"))
        plt.close()

# Function to perform inference and visualize Grad-CAM heatmaps
@torch.no_grad()
def perform_inference(test_loader, model, cfg):
    """
    Perform inference on the test set and visualize Grad-CAM heatmaps.
    Args:
        test_loader: DataLoader for the test set.
        model: SlowFast model.
        cfg: SlowFast configuration object.
    """
    layers_to_test = {
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

    for inputs, video_id, slow_frames, fast_frames, labels in test_loader:
        logger.info(f"Processing video: {video_id[0]}")
        logger.debug(f"Input slow shape: {inputs[0].shape}, fast shape: {inputs[1].shape}")
        inputs = [inp.cuda(non_blocking=True) for inp in inputs]

        # Extract and visualize Grad-CAM heatmaps from different layers
        for depth, layers in layers_to_test.items():
            target_layer_slow = layers['slow']
            target_layer_fast = layers['fast']
            heatmaps = extract_gradcam_heatmaps(model, target_layer_slow, target_layer_fast, inputs)
            logger.debug(f"{depth.capitalize()} - Slow Grad-CAM shape: {heatmaps['slow'].shape}")
            logger.debug(f"{depth.capitalize()} - Fast Grad-CAM shape: {heatmaps['fast'].shape}")
            gradcam_dir = os.path.join(cfg.OUTPUT_DIR, f"gradcam_{depth}")
            visualize_gradcam_heatmaps(heatmaps, slow_frames, fast_frames, video_id[0], gradcam_dir)

        # Extract features for the video
        model.eval()
        preds, feat = model(inputs)
        if cfg.NUM_GPUS > 1:
            preds, feat = du.all_gather([preds, feat])
        feat = feat.cpu().numpy()
        out_path = os.path.join(cfg.OUTPUT_DIR, "features")
        os.makedirs(out_path, exist_ok=True)
        out_file = f"{video_id[0]}_slowfast_features.npy"
        np.save(os.path.join(out_path, out_file), feat)
        logger.info(f"Saved features for {video_id[0]} to {os.path.join(out_path, out_file)}")
        del inputs, preds, feat
        torch.cuda.empty_cache()

# Function to fine-tune the model
def train(cfg, train_loader, val_loader, model):
    """
    Fine-tune the SlowFast model on the training set.
    Args:
        cfg: SlowFast configuration object.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        model: SlowFast model to fine-tune.
    Returns:
        model: Fine-tuned SlowFast model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Modify the head for your task (2 classes: 0 and 1)
    num_classes = 2
    model = modify_slowfast_head(model, num_classes, device)
    
    # Freeze early layers (up to s3)
    freeze_until = 's3'
    for name, param in model.named_parameters():
        if 's1' in name or 's2' in name or 's3' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    logger.info(f"Froze layers up to {freeze_until}, trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Define optimizer and loss
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, video_id, slow_frames, fast_frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = [inp.to(device) for inp in inputs]
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/train_total:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, video_id, slow_frames, fast_frames, labels in val_loader:
                inputs = [inp.to(device) for inp in inputs]
                labels = labels.to(device)
                
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "slowfast_finetuned.pt"))
    logger.info(f"Saved fine-tuned model to {os.path.join(cfg.OUTPUT_DIR, 'slowfast_finetuned.pt')}")
    return model

# Main function to run fine-tuning and inference
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
    cu.load_test_checkpoint(cfg, model)

    # Define paths
    pickle_dir = "D:/pickle_dir/fine_tune"
    bbox_dir = "D:/frcnn_bboxes/bboxes_top"
    pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))

    logger.info(f"Found {len(pickle_files)} pickle files to process.")
    logger.info("----------------------------------------------------------")

    # Collect all segments
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

    # Split into train and validation (80-20 split)
    train_segments, val_segments = train_test_split(all_segments, test_size=0.2, random_state=42)
    logger.info(f"Dataset split: {len(train_segments)} training segments, {len(val_segments)} validation segments")

    # Create datasets
    train_datasets = [SinglePickleFrameDataset(
        frames=seg['frames'],
        video_id=seg['video_id'],
        label=seg['label'],
        cfg=cfg,
        bbox_dir=bbox_dir
    ) for seg in train_segments]
    
    val_datasets = [SinglePickleFrameDataset(
        frames=seg['frames'],
        video_id=seg['video_id'],
        label=seg['label'],
        cfg=cfg,
        bbox_dir=bbox_dir
    ) for seg in val_segments]

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_datasets),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=False
    )

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(val_datasets),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=False
    )

    # Fine-tune the model
    logger.info("Starting fine-tuning...")
    model = train(cfg, train_loader, val_loader, model)

    # Perform inference on validation set to visualize Grad-CAM heatmaps
    logger.info("Performing inference on validation set...")
    perform_inference(val_loader, model, cfg)

    logger.info("----------------------------------------------------------")

def main():
    """
    Entry point for the script.
    """
    args = parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_files[0])
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == "__main__":
    main()